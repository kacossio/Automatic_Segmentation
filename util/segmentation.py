"""
Text-prompted image segmentation using Meta SAM 3.

SAM 3 promptable concept segmentation goes from text prompt straight to
instance masks in one model — one call per ontology prompt. Produces COCO
JSON output from per-image masks.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv
import torch
import yaml
from PIL import Image
from pycocotools import mask as coco_mask
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Classes that can describe the same physical person. SAM3 runs one text
# prompt per class, so a single person is often boxed twice under two of
# these (most commonly "player" + "referee", also "player" + "goalkeeper").
PERSON_CLASSES = ("player", "goalkeeper", "referee")
# IoU above which two person-class boxes are treated as the same object
# rather than two distinct, overlapping people. Matches verify.py's
# DUPLICATE_IOU so the deterministic pass here and the LLM fallback there
# agree on what counts as a duplicate.
CROSS_CLASS_DUP_IOU = 0.85


def _xyxy_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _feet_on_field(box: np.ndarray, field_mask: np.ndarray) -> bool:
    """True if the ground around an xyxy box's feet falls inside the field
    mask. Shared by resolve_person_duplicates and the referee singleton
    filter in _apply_domain_filters.

    Samples a ring of points around (not under) the box's bottom edge,
    pushed outward by a margin, rather than the single bottom-center pixel.
    The field mask is a segmentation of visible grass, so it has a
    person-shaped gap under every occluding body -- the bottom-center pixel
    of a tightly-fit box is the person's own foot/shoe, which is never
    "field", regardless of whether they're standing on the pitch or not.
    Sampled this checked false for a referee standing in the dead centre of
    open pitch grass before the ring fix. True if any ring point is field.
    """
    x1, y1, x2, y2 = box
    h, w = field_mask.shape
    bw = max(x2 - x1, 1.0)
    margin = max(bw * 0.4, 8.0)
    xs = (x1 - margin, (x1 + x2) / 2.0, x2 + margin)
    ys = (y2 - 2.0, y2 + margin)
    for yy in ys:
        for xx in xs:
            fx = min(max(int(round(xx)), 0), w - 1)
            fy = min(max(int(round(yy)), 0), h - 1)
            if field_mask[fy, fx]:
                return True
    return False


def resolve_person_duplicates(
    xyxy: np.ndarray,
    class_names: List[str],
    confidence: np.ndarray,
    field_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean keep-mask resolving same-object cross-class person duplicates.

    When two person-class boxes (player/goalkeeper/referee) overlap above
    CROSS_CLASS_DUP_IOU, they're almost always the same person caught by two
    of the per-class text prompts, not two people. Resolved structurally
    rather than by confidence: class_thresholds sets very different floors
    per class on purpose (referee: 0.15 vs player: 0.30, to keep rare/
    low-confidence referees for review), so a referee box legitimately
    scoring lower than the player box on the very same person is expected,
    not a sign the player label is right. The specific class (referee/
    goalkeeper) wins over the generic "player" class on the same object --
    unless a field mask is available and that specific-class box's feet
    fall off the pitch, in which case an off-field "referee"/"goalkeeper" is
    as likely to be a misfire (bench staff, fourth official) as a real one,
    so it falls back to confidence instead of automatically winning.
    Referee-vs-goalkeeper (no class-name signal either way) always falls
    back to confidence. Shared by Segmenter._apply_domain_filters
    (per-frame, at label time) and dedup_annotations.py (one-off pass over
    an existing annotations.json), so both apply the exact same rule.
    """
    keep = np.ones(len(class_names), dtype=bool)
    person_idx = [i for i, n in enumerate(class_names) if n in PERSON_CLASSES]
    for a in range(len(person_idx)):
        i = person_idx[a]
        for b in range(a + 1, len(person_idx)):
            j = person_idx[b]
            if not keep[i] or not keep[j]:
                continue
            if _xyxy_iou(xyxy[i], xyxy[j]) < CROSS_CLASS_DUP_IOU:
                continue
            name_i, name_j = class_names[i], class_names[j]
            player, other = None, None
            if name_i == "player" and name_j != "player":
                player, other = i, j
            elif name_j == "player" and name_i != "player":
                player, other = j, i

            if player is not None and (
                field_mask is None or _feet_on_field(xyxy[other], field_mask)
            ):
                keep[player] = False
            else:
                loser = i if confidence[i] < confidence[j] else j
                keep[loser] = False
    return keep


class Segmenter:

    def __init__(self, config: str):
        with open(config) as f:
            cfg = yaml.safe_load(f)

        self.source_directory: str = cfg["source_directory"]
        self.dest_directory: str = cfg["dest_directory"]
        self.ontology_map: Dict[str, str] = cfg["ontology"]

        self.sam3_model_id: str = cfg.get("sam3_model", "facebook/sam3")
        # Published defaults from the SAM 3 model card.
        self.sam3_score_threshold: float = float(cfg.get("sam3_score_threshold", 0.5))
        self.sam3_mask_threshold: float = float(cfg.get("sam3_mask_threshold", 0.5))
        # Per-class confidence floor. A class listed here REPLACES
        # sam3_score_threshold for that class (it does not stack on top of it);
        # only classes absent from this map fall back to the global threshold.
        self.class_thresholds: Dict[str, float] = {
            str(k): float(v) for k, v in cfg.get("class_thresholds", {}).items()
        }
        # When false (default) nothing is dropped before human review: the
        # "one ball in play" rule becomes the reviewer's decision.
        self.apply_domain_filters: bool = bool(cfg.get("apply_domain_filters", False))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.classes: List[str] = list(dict.fromkeys(self.ontology_map.values()))
        # Map normalized prompt -> class id.
        self.prompt_to_class_id: Dict[str, int] = {
            p.lower().strip(): self.classes.index(c) for p, c in self.ontology_map.items()
        }

        # (width, height) captured during predict_image so _write_coco never
        # re-reads the source images just to learn their size.
        self._image_wh: Dict[str, Tuple[int, int]] = {}
        # image path -> already-built COCO annotation dicts (sans id/image_id).
        # This is the ONLY per-frame state kept across the run: detections are
        # RLE-encoded into these dicts immediately after prediction and the
        # sv.Detections (with its full-resolution boolean masks, ~MBs per
        # detection) is dropped, so memory stays flat over 100k-frame runs.
        self._frames: Dict[str, List[dict]] = {}

        os.makedirs(self.dest_directory, exist_ok=True)

    def load_models(self):
        processor = Sam3Processor.from_pretrained(self.sam3_model_id)
        model = (
            Sam3Model.from_pretrained(self.sam3_model_id)
            .to(self.device)
            .eval()
        )
        return processor, model

    def predict_image(self, image_path: str, processor, model) -> sv.Detections:
        det = self._predict_sam3(image_path, processor, model)
        return self._apply_domain_filters(det) if self.apply_domain_filters else det

    def _predict_sam3(self, image_path: str, processor, model) -> sv.Detections:
        """SAM 3 promptable concept segmentation, one call per ontology prompt.

        Mirrors the model card's documented usage:
        ``processor(images=pil, text=prompt) -> model(**inputs) ->
        post_process_instance_segmentation``. No vision-feature caching, no
        bf16 autocast, no prefetch -- the model card pattern was already fast
        on this dataset, and extra plumbing tended to make it slower.
        """
        pil_image = Image.open(image_path).convert("RGB")
        w, h = pil_image.size
        self._image_wh[image_path] = (w, h)

        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_masks: List[np.ndarray] = []
        all_class_ids: List[np.ndarray] = []

        for prompt, class_id in self.prompt_to_class_id.items():
            # Per-class confidence floor on top of sam3_score_threshold; classes
            # absent from class_thresholds fall back to the global SAM3 floor.
            floor = self.class_thresholds.get(
                self.classes[class_id], self.sam3_score_threshold
            )
            inputs = processor(
                images=pil_image, text=prompt, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                outputs = model(**inputs)
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=floor,
                mask_threshold=self.sam3_mask_threshold,
                target_sizes=[(h, w)],
            )[0]
            if len(results["boxes"]) == 0:
                continue
            all_boxes.append(results["boxes"].cpu().numpy())
            all_scores.append(results["scores"].cpu().numpy())
            all_masks.append(results["masks"].cpu().numpy().astype(bool))
            all_class_ids.append(np.full(len(results["boxes"]), class_id, dtype=int))

        if not all_boxes:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.concatenate(all_boxes, axis=0).astype(float),
            class_id=np.concatenate(all_class_ids, axis=0),
            confidence=np.concatenate(all_scores, axis=0).astype(float),
            mask=np.concatenate(all_masks, axis=0),
        )

    def _apply_domain_filters(self, det: sv.Detections) -> sv.Detections:
        """Soccer-specific post-filter for single-instance classes.

        - Cross-class person duplicates: when two person-class boxes (player/
          goalkeeper/referee) overlap heavily, they're almost always the same
          person caught by two of the per-class text prompts, not two people.
          Resolved structurally rather than by confidence: class_thresholds
          sets very different floors per class on purpose (referee: 0.15 vs
          player: 0.30, to keep rare/low-confidence referees for review), so
          a referee box legitimately scoring lower than the player box on the
          very same person is expected, not a sign the player label is right.
          The specific class (referee/goalkeeper) wins over the generic
          "player" class on the same object, unless it's off the detected
          field (see resolve_person_duplicates' docstring).
        - Ball: reject player-shaped boxes (SAM3 sometimes grounds "soccer
          ball" onto the whole ball-carrying player), then keep only the
          highest-confidence remaining detection (one ball in play). If none
          are ball-shaped, keep none rather than a player.
        - Referee: prefer referees standing inside the detected field (feet =
          bottom-centre of the box), then keep the single highest-confidence
          one. If none are on the field, or no field was detected this frame,
          fall back to the global highest-confidence referee so a bad/missing
          field mask never drops the only referee (review still catches it).
        """
        if len(det) == 0:
            return det

        cls_index = {name: i for i, name in enumerate(self.classes)}

        field_mask = None
        field_id = cls_index.get("field")
        if field_id is not None and det.mask is not None:
            field_idx = np.where(det.class_id == field_id)[0]
            if len(field_idx) > 0:
                field_mask = np.any(det.mask[field_idx], axis=0)

        class_names = [self.classes[c] for c in det.class_id]
        keep = resolve_person_duplicates(
            det.xyxy, class_names, det.confidence, field_mask
        )

        ball_id = cls_index.get("ball")
        if ball_id is not None:
            idx = np.where(det.class_id == ball_id)[0]
            if len(idx) > 0:
                # A real ball box is small and roughly square. The model
                # sometimes grounds the "soccer ball" prompt onto the whole
                # player who has the ball, producing a tall, large box. Reject
                # those before the single-ball dedup so the kept ball is never a
                # player; if nothing is ball-shaped, keep no ball this frame
                # (the reviewer can flag the miss).
                frame_area = None
                if det.mask is not None:
                    mh, mw = det.mask.shape[1:]
                    frame_area = float(mh * mw)
                plausible = []
                for i in idx:
                    x1, y1, x2, y2 = det.xyxy[i]
                    bw, bh = float(x2 - x1), float(y2 - y1)
                    if bw <= 0 or bh <= 0:
                        continue
                    tall = bh > 1.6 * bw  # players are markedly taller than wide
                    too_big = (
                        frame_area is not None
                        and (bw * bh) / frame_area > 0.03
                    )
                    if not tall and not too_big:
                        plausible.append(i)
                keep[idx] = False
                if plausible:
                    cand = np.array(plausible)
                    keep[cand[int(np.argmax(det.confidence[cand]))]] = True

        ref_id = cls_index.get("referee")
        if ref_id is not None:
            idx = np.where(det.class_id == ref_id)[0]
            if len(idx) > 1:
                candidates = idx
                if field_mask is not None:
                    on_field = [i for i in idx if _feet_on_field(det.xyxy[i], field_mask)]
                    if on_field:
                        candidates = np.array(on_field)
                best = candidates[int(np.argmax(det.confidence[candidates]))]
                keep[idx] = False
                keep[best] = True

        return det[keep]

    def _collect_images(self) -> List[str]:
        root = Path(self.source_directory)
        paths: List[str] = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(str(p) for p in root.glob(f"*{ext}"))
        return sorted(paths)

    def _detections_to_coco(self, det: sv.Detections) -> List[dict]:
        """One image's sv.Detections -> COCO annotation dicts (no id/image_id).

        Masks are compressed-RLE encoded (pycocotools), the same lossless
        encoding review.py emits and that render_overlays / review.py already
        decode -- avoiding supervision's slower, lossy polygon approximation.
        Each detection's confidence is attached inline as ``score`` (bound to
        its detection at creation, so no fragile post-hoc index matching).
        """
        if det.confidence is None or len(det) == 0:
            return []
        out: List[dict] = []
        for i in range(len(det)):
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[i])
            mask = det.mask[i].astype(np.uint8)
            rle = coco_mask.encode(np.asfortranarray(mask))
            out.append(
                {
                    "category_id": int(det.class_id[i]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": int(mask.sum()),
                    "segmentation": {
                        "size": [int(mask.shape[0]), int(mask.shape[1])],
                        "counts": rle["counts"].decode("ascii"),
                    },
                    "iscrowd": 0,
                    "score": float(det.confidence[i]),
                }
            )
        return out

    def _image_size(self, path: str) -> Tuple[int, int]:
        """(width, height) for an image, preferring the size captured during
        prediction; falls back to a header-only read for failed/empty images
        (PIL does not decode pixels for ``.size``). An unreadable image (the
        usual reason its prediction failed) records (0, 0) rather than
        aborting the run a second time."""
        wh = self._image_wh.get(path)
        if wh is not None:
            return wh
        try:
            with Image.open(path) as im:
                wh = im.size
        except Exception:
            wh = (0, 0)
        self._image_wh[path] = wh
        return wh

    @property
    def _partial_path(self) -> Path:
        """Append-only per-frame checkpoint (one JSON line per frame)."""
        return Path(self.dest_directory) / "annotations.partial.jsonl"

    def _load_partial(self) -> int:
        """Resume state from annotations.partial.jsonl into self._frames.

        Returns the number of frames loaded. Malformed trailing lines (a run
        killed mid-write) are skipped so a partial last line never blocks
        resuming. Delete the file to force a full relabel.
        """
        if not self._partial_path.exists():
            return 0
        loaded = 0
        with open(self._partial_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated final line from a killed run
                path = rec["path"]
                self._frames[path] = rec["annotations"]
                self._image_wh[path] = (rec["width"], rec["height"])
                loaded += 1
        return loaded

    def _append_partial(self, f, path: str, anns: List[dict]):
        w, h = self._image_size(path)
        f.write(
            json.dumps(
                {
                    "path": path,
                    "file_name": Path(path).name,
                    "width": w,
                    "height": h,
                    "annotations": anns,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        f.flush()

    def _write_coco(
        self, frames: Dict[str, List[dict]], progress: bool = True
    ) -> Path:
        """Build COCO from the per-frame annotation dicts and write
        annotations.json directly (no supervision DetectionDataset / image
        re-read / double serialization).

        Called once at the end of a run; mid-run durability comes from the
        per-frame JSONL checkpoint (annotations.partial.jsonl), so a killed
        run resumes instead of rewriting the whole JSON every N frames.
        """
        licenses = [
            {
                "id": 1,
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "name": "CC BY 4.0",
            }
        ]
        categories = [
            {"id": i, "name": n, "supercategory": "common-objects"}
            for i, n in enumerate(self.classes)
        ]
        captured = datetime.now().strftime("%m/%d/%Y,%H:%M:%S")

        coco_images: List[dict] = []
        coco_annotations: List[dict] = []
        image_id, annotation_id = 1, 1

        items = sorted(frames.items())
        if progress:
            items = tqdm(items, desc="Writing COCO", leave=False, unit="img")
        for path, anns in items:
            w, h = self._image_size(path)
            coco_images.append(
                {
                    "id": image_id,
                    "license": 1,
                    "file_name": Path(path).name,
                    "height": h,
                    "width": w,
                    "date_captured": captured,
                }
            )
            for a in anns:
                coco_annotations.append(
                    {"id": annotation_id, "image_id": image_id, **a}
                )
                annotation_id += 1
            image_id += 1

        coco = {
            "info": {},
            "licenses": licenses,
            "categories": categories,
            "images": coco_images,
            "annotations": coco_annotations,
        }

        coco_path = Path(self.dest_directory) / "annotations.json"
        with open(coco_path, "w") as f:
            json.dump(coco, f, separators=(",", ":"))
        return coco_path

    def run(self):
        image_paths = self._collect_images()
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.source_directory}")
        print(f"Found {len(image_paths)} image(s) in '{self.source_directory}'.")

        resumed = self._load_partial()
        if resumed:
            print(
                f"Resuming: {resumed} frame(s) already in {self._partial_path} "
                "will be skipped (delete the file to relabel from scratch)."
            )
        todo = [p for p in image_paths if p not in self._frames]

        failures: List[str] = []
        if todo:
            print(f"Loading SAM 3 ({self.sam3_model_id}) on {self.device}...")
            processor, model = self.load_models()

            with open(self._partial_path, "a") as partial:
                for path in tqdm(todo, desc="Segmenting"):
                    try:
                        det = self.predict_image(path, processor, model)
                        # RLE-encode now and drop the Detections (and its
                        # full-res masks) so memory stays flat over the run.
                        anns = self._detections_to_coco(det)
                    except Exception as e:  # one bad image must not abort the batch
                        anns = []
                        failures.append(path)
                        tqdm.write(f"FAILED {path}: {type(e).__name__}: {e}")
                    self._frames[path] = anns
                    self._append_partial(partial, path, anns)

        coco_path = self._write_coco(self._frames)
        print(f"Done. COCO annotations written to: {coco_path}")
        if failures:
            print(f"\n{len(failures)} image(s) failed and were written empty:")
            for p in failures:
                print(f"  - {p}")
