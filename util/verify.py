"""Automated annotation verification using Claude (Anthropic Batch API).

Does per-detection what a human does in review.py -- keep / reject /
reclassify -- for every detection in <dest_directory>/annotations.json.
Confident verdicts are auto-resolved; anything low-confidence or otherwise
unusable is left flagged for the human pass in the existing review.py UI.

Verdicts are constrained via structured outputs (output_config.format), so a
malformed/unparseable response is not a failure mode this code needs to
handle -- the residual failure modes are a request that didn't produce a
response at all (error/refusal) and a schema-valid response whose length
doesn't match the frame's detection count (array length isn't expressible
in the schema; checked post-hoc).

review.py is subtractive-only (it never adds a missed detection), so this
pass cannot lower recall -- it can only change how much precision review.py
needs to restore, and how much of that restoration is already done.

Usage:
    python verify.py --config config.yaml
    python verify.py --config config.yaml --sync --limit 3   # smoke test
"""
import base64
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
import cv2
import yaml
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from tqdm import tqdm

# Numbered-box overlay color (BGR) -- distinct from review.py's selection
# highlight colors so verification overlays aren't mistaken for UI state.
BOX_COLOR = (0, 140, 255)

BATCH_POLL_SECONDS = 30

SYSTEM_PROMPT_TEMPLATE = """You are performing quality control on auto-generated object detection labels, to prepare them for human review.

The auto-labeler over-generates: false positives (wrong boxes, wrong classes) are expected and common. Your job is per-detection quality control, not detection -- you may only keep, reject, or reclassify the numbered detections already boxed in each image. Never suggest an object that isn't already boxed; missed detections are out of scope and are handled separately.

Ontology classes: {classes}

Return one verdict per numbered detection, in the same order as the numbered detections. For each:
- "keep": true if the box is on the right kind of object and worth keeping, false if it should be rejected (wrong object, spurious box, duplicate).
- "class": the correct class name from the ontology list above (repeat the current class if you agree with it).
- "confidence": "high" if you are confident in this keep/class decision, "low" if you are at all unsure. A human reviewer double-checks anything marked "low", so use it liberally whenever the crop is ambiguous, occluded, or borderline."""

_EMPTY_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
}


def annotate_numbered(image, anns: List[dict], cats: Dict[int, str]):
    """Draw each detection's bbox with an "{index}:{class name}" label.

    Reads bboxes directly (no RLE mask decode) -- cheaper than
    render_overlays.annotate_detections at 100k-frame scale, since the
    verifier only needs box + class, not the mask.
    """
    out = image.copy()
    for i, a in enumerate(anns):
        x, y, w, h = a["bbox"]
        x1, y1, x2, y2 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)

        label = f"{i}:{cats[a['category_id']]}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty), BOX_COLOR, -1)
        cv2.putText(
            out, label, (x1 + 2, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


def _encode_jpeg_b64(image, max_width: int) -> str:
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(
            image, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode overlay image")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


class Verifier:

    def __init__(self, config: str):
        with open(config) as f:
            cfg = yaml.safe_load(f)

        self.source_directory: str = cfg["source_directory"]
        self.dest_directory: str = cfg["dest_directory"]

        verify_cfg = cfg.get("verify", {})
        self.model: str = verify_cfg.get("model", "claude-sonnet-5")
        self.use_batch: bool = bool(verify_cfg.get("use_batch", True))
        self.max_image_width: int = int(verify_cfg.get("max_image_width", 1280))
        self.max_tokens: int = int(verify_cfg.get("max_tokens", 1024))

        self.client = anthropic.Anthropic()

        coco_path = Path(self.dest_directory) / "annotations.json"
        with open(coco_path) as f:
            self.coco: dict = json.load(f)

        self.cats: Dict[int, str] = {c["id"]: c["name"] for c in self.coco["categories"]}
        self.name_to_id: Dict[str, int] = {n: i for i, n in self.cats.items()}
        self.per_img: Dict[int, List[dict]] = {}
        for a in self.coco["annotations"]:
            self.per_img.setdefault(a["image_id"], []).append(a)

        # Shared across every request in a run; the ephemeral cache_control
        # lets the Batch API cache it across the batch (images are per-frame
        # and don't cache).
        classes = sorted(set(self.cats.values()))
        self.system_prompt = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT_TEMPLATE.format(classes=", ".join(classes)),
                "cache_control": {"type": "ephemeral"},
            }
        ]

        # Structured outputs: constrains "class" to the actual ontology and
        # guarantees valid, parseable JSON -- the only thing schema can't
        # express is the array length (unsupported "complex array
        # constraint"), so that's still checked post-hoc in _parse_verdicts.
        # Same schema for every frame regardless of detection count, so it's
        # wrapped in an object per the API's documented json_schema shape
        # rather than a bare array at the root.
        self.output_config = {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "verdicts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "keep": {"type": "boolean"},
                                    "class": {"type": "string", "enum": classes},
                                    "confidence": {"type": "string", "enum": ["high", "low"]},
                                },
                                "required": ["keep", "class", "confidence"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["verdicts"],
                    "additionalProperties": False,
                },
            }
        }

    def _build_message(self, image_path: str, anns: List[dict]) -> dict:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        overlay = annotate_numbered(image, anns, self.cats)
        b64 = _encode_jpeg_b64(overlay, self.max_image_width)
        lines = [
            f"Detection {i}: currently '{self.cats[a['category_id']]}'"
            for i, a in enumerate(anns)
        ]
        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                },
                {"type": "text", "text": "\n".join(lines)},
            ],
        }

    def _parse_verdicts(
        self, text: Optional[str], n: int
    ) -> Tuple[Optional[List[dict]], Optional[str]]:
        """Raw response text -> (verdicts, frame_reason).

        ``frame_reason`` is set (and verdicts is None) whenever the whole
        frame's response can't be used, so every detection in it falls back
        to the default decision and gets flagged with the same reason.
        Structured outputs guarantees valid JSON matching the item schema,
        so a parse failure here means the request itself didn't produce a
        usable response (missing/refused) or the model returned the wrong
        number of verdicts (array length isn't schema-enforceable).
        """
        if not text:
            # Empty string covers a successful response with no text block
            # (e.g. a refusal) the same way None covers a request that never
            # produced a response at all (batch error/image not found).
            return None, "missing_response"
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None, "unparseable_response"
        if not isinstance(data, dict) or not isinstance(data.get("verdicts"), list):
            return None, "unexpected_response_shape"
        verdicts = data["verdicts"]
        if len(verdicts) != n:
            return None, "verdict_count_mismatch"
        return verdicts, None

    def _resolve_verdict(self, ann: dict, v) -> Tuple[bool, int, bool, Optional[str]]:
        """One detection's verdict -> (keep, class_id, confident, reason).

        ``reason`` is None when confident; otherwise it's why this detection
        is flagged. Falls back to the labeler's own keep=True/class-unchanged
        default whenever the verdict itself is unusable, matching review.py's
        inclusive default -- an unresolved detection is never silently
        dropped, only flagged for the human pass.
        """
        default_keep, default_cid = True, ann["category_id"]
        if not isinstance(v, dict):
            return default_keep, default_cid, False, "invalid_detection_verdict"

        keep = v.get("keep")
        cls = v.get("class")
        if not isinstance(keep, bool) or cls not in self.name_to_id:
            return default_keep, default_cid, False, "invalid_detection_verdict"

        if v.get("confidence") != "high":
            return keep, self.name_to_id[cls], False, "low_confidence"
        return keep, self.name_to_id[cls], True, None

    @staticmethod
    def _usage_from_message(msg) -> dict:
        return {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
            "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
        }

    @staticmethod
    def _accumulate(total: dict, u: dict):
        for k in total:
            total[k] += u.get(k, 0)

    def _run_sync(self, images: List[dict]) -> Tuple[Dict[int, Optional[str]], dict]:
        results: Dict[int, Optional[str]] = {}
        usage = dict(_EMPTY_USAGE)
        for img in tqdm(images, desc="Verifying", unit="img"):
            image_id = img["id"]
            anns = self.per_img.get(image_id, [])
            if not anns:
                results[image_id] = None
                continue
            image_path = str(Path(self.source_directory) / img["file_name"])
            try:
                message = self._build_message(image_path, anns)
            except FileNotFoundError:
                results[image_id] = None
                continue

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[message],
                output_config=self.output_config,
            )
            results[image_id] = "".join(
                b.text for b in response.content if b.type == "text"
            )
            self._accumulate(usage, self._usage_from_message(response))
        return results, usage

    def _run_batch(self, images: List[dict]) -> Tuple[Dict[int, Optional[str]], dict]:
        requests = []
        results: Dict[int, Optional[str]] = {}
        for img in images:
            image_id = img["id"]
            anns = self.per_img.get(image_id, [])
            if not anns:
                results[image_id] = None
                continue
            image_path = str(Path(self.source_directory) / img["file_name"])
            try:
                message = self._build_message(image_path, anns)
            except FileNotFoundError:
                results[image_id] = None
                continue
            requests.append(
                Request(
                    custom_id=str(image_id),
                    params=MessageCreateParamsNonStreaming(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        system=self.system_prompt,
                        messages=[message],
                        output_config=self.output_config,
                    ),
                )
            )

        usage = dict(_EMPTY_USAGE)
        if not requests:
            return results, usage

        batch = self.client.messages.batches.create(requests=requests)
        print(f"Submitted batch {batch.id} ({len(requests)} request(s)); polling...")
        while True:
            batch = self.client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            time.sleep(BATCH_POLL_SECONDS)
        print(
            f"Batch {batch.id} ended "
            f"(succeeded={batch.request_counts.succeeded}, errored={batch.request_counts.errored})."
        )

        # Results arrive keyed by custom_id, not in submission order.
        for result in self.client.messages.batches.results(batch.id):
            image_id = int(result.custom_id)
            if result.result.type == "succeeded":
                msg = result.result.message
                results[image_id] = "".join(
                    b.text for b in msg.content if b.type == "text"
                )
                self._accumulate(usage, self._usage_from_message(msg))
            else:
                results[image_id] = None
        return results, usage

    def _resolve_and_write(
        self, images: List[dict], results: Dict[int, Optional[str]], usage: dict
    ) -> dict:
        decisions: Dict[str, dict] = {}
        confident_keys: set = set()
        reviewed: List[int] = []
        flag_reasons: Dict[str, int] = {}
        counts = {"auto_kept": 0, "auto_rejected": 0, "reclassified": 0, "flagged": 0}

        for img in images:
            image_id = img["id"]
            anns = self.per_img.get(image_id, [])
            if not anns:
                reviewed.append(image_id)
                continue

            verdicts, frame_reason = self._parse_verdicts(results.get(image_id), len(anns))
            frame_confident = verdicts is not None

            for i, a in enumerate(anns):
                key = f"{image_id}:{a['id']}"
                v = verdicts[i] if verdicts is not None else None
                keep, class_id, confident, per_ann_reason = self._resolve_verdict(a, v)
                decisions[key] = {"keep": keep, "class_id": class_id}

                if not confident:
                    frame_confident = False
                    reason = frame_reason or per_ann_reason
                    flag_reasons[reason] = flag_reasons.get(reason, 0) + 1
                    counts["flagged"] += 1
                    continue

                confident_keys.add(key)
                if not keep:
                    counts["auto_rejected"] += 1
                elif class_id != a["category_id"]:
                    counts["reclassified"] += 1
                else:
                    counts["auto_kept"] += 1

            if frame_confident:
                reviewed.append(image_id)

        self._write_review_state(decisions, reviewed)
        reviewed_path, kept_n = self._write_reviewed_annotations(decisions, confident_keys)
        return self._write_report(counts, flag_reasons, usage, len(images), kept_n, reviewed_path)

    def _write_review_state(self, decisions: Dict[str, dict], reviewed: List[int]):
        path = Path(self.dest_directory) / "review_state.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "decisions": decisions,
                    "reviewed": sorted(set(reviewed)),
                    "added": {},
                    "next_added_id": -1,
                },
                f,
                indent=2,
            )

    def _write_reviewed_annotations(
        self, decisions: Dict[str, dict], confident_keys: set
    ) -> Tuple[Path, int]:
        """Kept-only standard COCO -- confident keeps/reclassifies only.

        Same shape as review.py's save_reviewed() output, so a fresh run of
        this pass with zero flags is already a valid training artifact.
        """
        out_anns = []
        for img_id, anns in self.per_img.items():
            for a in anns:
                key = f"{img_id}:{a['id']}"
                if key not in confident_keys:
                    continue
                d = decisions[key]
                if not d["keep"]:
                    continue
                na = dict(a)
                na["category_id"] = d["class_id"]
                out_anns.append(na)

        out = {
            "images": self.coco["images"],
            "categories": self.coco["categories"],
            "annotations": out_anns,
        }
        for opt in ("info", "licenses"):
            if opt in self.coco:
                out[opt] = self.coco[opt]

        path = Path(self.dest_directory) / "reviewed_annotations.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        return path, len(out_anns)

    def _write_report(
        self, counts: dict, flag_reasons: dict, usage: dict,
        n_images: int, kept_n: int, reviewed_path: Path,
    ) -> dict:
        report = {
            "model": self.model,
            "images_processed": n_images,
            "counts": counts,
            "flag_reasons": flag_reasons,
            "usage": usage,
            "reviewed_annotations_path": str(reviewed_path),
            "reviewed_annotations_kept": kept_n,
        }
        path = Path(self.dest_directory) / "verification_report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        return report

    def run(self, limit: Optional[int] = None, sync: bool = False) -> dict:
        images = list(self.coco["images"])
        if limit is not None:
            images = images[:limit]

        if sync or not self.use_batch:
            results, usage = self._run_sync(images)
        else:
            results, usage = self._run_batch(images)

        return self._resolve_and_write(images, results, usage)
