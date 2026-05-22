"""
Text-prompted image segmentation using HF Grounding DINO + SAM2.
Produces COCO JSON output from per-image masks.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv
import torch
import yaml
from PIL import Image
from pycocotools import mask as coco_mask
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Boxes Grounding DINO finds but cannot be unambiguously mapped to an ontology
# class are kept under this class for the human reviewer to reclassify or
# reject, rather than being silently dropped (which caps recall).
UNKNOWN_CLASS = "unknown"


class Segmenter:

    def __init__(self, config: str):
        with open(config) as f:
            cfg = yaml.safe_load(f)

        self.source_directory: str = cfg["source_directory"]
        self.dest_directory: str = cfg["dest_directory"]
        self.ontology_map: Dict[str, str] = cfg["ontology"]

        self.dino_model_id: str = cfg.get("dino_model", "IDEA-Research/grounding-dino-tiny")
        self.sam2_model: str = cfg.get("sam2_model", "facebook/sam2-hiera-large")
        # Recall-first global floor: keep almost everything from DINO, then
        # recover precision per class and via human review.
        self.box_threshold: float = float(cfg.get("box_threshold", 0.15))
        self.text_threshold: float = float(cfg.get("text_threshold", 0.15))
        # Per-class confidence floor applied *after* label->class mapping.
        # Classes absent here fall back to box_threshold.
        self.class_thresholds: Dict[str, float] = {
            str(k): float(v) for k, v in cfg.get("class_thresholds", {}).items()
        }
        # Class-agnostic NMS IoU on DINO boxes before SAM2.
        self.nms_iou: float = float(cfg.get("nms_iou", 0.80))
        # When false (default) nothing is dropped before human review: the
        # "one ball in play" rule becomes the reviewer's decision.
        self.apply_domain_filters: bool = bool(cfg.get("apply_domain_filters", False))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.prompts: List[str] = [p.lower().strip() for p in self.ontology_map.keys()]
        self.classes: List[str] = list(dict.fromkeys(self.ontology_map.values()))
        # Map normalized prompt -> class id. Built before UNKNOWN is appended so
        # these indices stay valid.
        self.prompt_to_class_id: Dict[str, int] = {
            p.lower().strip(): self.classes.index(c) for p, c in self.ontology_map.items()
        }
        # UNKNOWN appended last so the indices above are unchanged.
        if UNKNOWN_CLASS not in self.classes:
            self.classes.append(UNKNOWN_CLASS)
        self.unknown_class_id: int = self.classes.index(UNKNOWN_CLASS)

        # (width, height) captured during predict_image so _write_coco never
        # re-reads the source images just to learn their size.
        self._image_wh: Dict[str, Tuple[int, int]] = {}
        # image path -> already-built COCO annotation dicts (sans id/image_id).
        # Each mask is RLE-encoded exactly once; checkpoint writes reuse this
        # instead of re-encoding every image processed so far (was O(N^2)).
        self._coco_cache: Dict[str, List[dict]] = {}

        os.makedirs(self.dest_directory, exist_ok=True)

    def load_models(self):
        processor = AutoProcessor.from_pretrained(self.dino_model_id)
        dino = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self.dino_model_id)
            .to(self.device)
            .eval()
        )
        sam = SAM2ImagePredictor.from_pretrained(self.sam2_model, device=self.device)
        return processor, dino, sam

    def _label_to_class_id(self, label: str) -> Optional[int]:
        """Map a Grounding DINO text label to an ontology class id.

        Grounding DINO grounds boxes to *token spans*, so for a multi-word
        prompt it usually returns a sub-phrase (e.g. "player" for the prompt
        "soccer player") and drops words that are ambiguous across prompts
        (e.g. "soccer", which is shared by player/ball/field). Matching must
        therefore be word-level and bidirectional, not full-phrase exact.
        """
        label = label.lower().strip()
        if not label:
            return None

        # 1. Exact phrase match.
        if label in self.prompt_to_class_id:
            return self.prompt_to_class_id[label]

        label_words = set(re.findall(r"[a-z]+", label))
        if not label_words:
            return None

        # 2. Word-overlap scoring. The best prompt is the one sharing the most
        #    words with the label; a tie between *different* classes means the
        #    label is ambiguous (e.g. bare "soccer") -> drop rather than guess.
        def words_match(a: str, b: str) -> bool:
            # Exact, or substring-either-direction for plural / affixed
            # variants ("players" vs "player"). Length guard avoids noise
            # like matching "a" inside "ball".
            if a == b:
                return True
            return min(len(a), len(b)) >= 4 and (a in b or b in a)

        best_cid: Optional[int] = None
        best_score = 0
        ambiguous = False
        for p, cid in self.prompt_to_class_id.items():
            prompt_words = set(re.findall(r"[a-z]+", p))
            score = sum(
                any(words_match(lw, pw) for pw in prompt_words)
                for lw in label_words
            )
            if score > best_score:
                best_score, best_cid, ambiguous = score, cid, False
            elif score == best_score and score > 0 and cid != best_cid:
                ambiguous = True

        if best_score == 0 or ambiguous:
            return None
        return best_cid

    def predict_image(self, image_path: str, processor, dino, sam) -> sv.Detections:
        pil_image = Image.open(image_path).convert("RGB")
        w, h = pil_image.size
        self._image_wh[image_path] = (w, h)

        inputs = processor(
            images=pil_image,
            text=[self.prompts],
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = dino(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
            text_labels=[self.prompts],
        )[0]

        boxes = results["boxes"].cpu().numpy()  # xyxy absolute
        scores = results["scores"].cpu().numpy()
        text_labels = results["text_labels"]

        if len(boxes) == 0:
            return sv.Detections.empty()

        # Label -> class. Unmappable / ambiguous boxes are *kept* as UNKNOWN
        # (the reviewer reclassifies or rejects them) rather than dropped, so
        # recall is not capped by DINO's grounding ambiguity.
        class_ids = np.array(
            [
                cid if (cid := self._label_to_class_id(lbl)) is not None
                else self.unknown_class_id
                for lbl in text_labels
            ],
            dtype=int,
        )

        # Per-class confidence floor (the global box_threshold is recall-first
        # low; this restores precision per class).
        floors = np.array(
            [
                self.class_thresholds.get(self.classes[c], self.box_threshold)
                for c in class_ids
            ],
            dtype=float,
        )
        keep = scores >= floors
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        if len(boxes) == 0:
            return sv.Detections.empty()

        # Class-agnostic NMS drops near-duplicate boxes (e.g. "soccer player"
        # vs "goalkeeper" on the same person) before paying SAM2 cost. High IoU
        # so a small ball box inside a large player box is not merged.
        det = sv.Detections(
            xyxy=boxes,
            class_id=class_ids,
            confidence=scores,
        ).with_nms(threshold=self.nms_iou, class_agnostic=True)
        if len(det) == 0:
            return sv.Detections.empty()

        image_rgb = np.asarray(pil_image)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            sam.set_image(image_rgb)
            masks_out: List[np.ndarray] = []
            for box in det.xyxy:
                # multimask_output=True returns 3 candidate masks at different
                # granularities; pick the highest-scoring (tighter on thin
                # limbs / occlusions). With False this argmax was dead code.
                masks, mask_scores, _ = sam.predict(box=box, multimask_output=True)
                masks_out.append(masks[int(np.argmax(mask_scores))].astype(bool))

        det.mask = np.stack(masks_out, axis=0)
        return self._apply_domain_filters(det) if self.apply_domain_filters else det

    def _apply_domain_filters(self, det: sv.Detections) -> sv.Detections:
        """Soccer-specific post-filter for single-instance classes.

        - Ball: reject player-shaped boxes (DINO often grounds "soccer ball"
          onto the whole ball-carrying player), then keep only the
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
        keep = np.ones(len(det), dtype=bool)

        ball_id = cls_index.get("ball")
        if ball_id is not None:
            idx = np.where(det.class_id == ball_id)[0]
            if len(idx) > 0:
                # A real ball box is small and roughly square. DINO frequently
                # grounds the "soccer ball" prompt onto the whole player who
                # has the ball, producing a tall, large box. Reject those
                # before the single-ball dedup so the kept ball is never a
                # player; if nothing is ball-shaped, keep no ball this frame
                # (the reviewer can click-add the real one).
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
                field_id = cls_index.get("field")
                if field_id is not None and det.mask is not None:
                    field_idx = np.where(det.class_id == field_id)[0]
                    if len(field_idx) > 0:
                        field_mask = np.any(det.mask[field_idx], axis=0)
                        h, w = field_mask.shape
                        on_field = []
                        for i in idx:
                            x1, _, x2, y2 = det.xyxy[i]
                            fx = min(max(int(round((x1 + x2) / 2)), 0), w - 1)
                            fy = min(max(int(round(y2)), 0), h - 1)
                            if field_mask[fy, fx]:
                                on_field.append(i)
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
        encoding review.py emits for human-added masks and that
        render_overlays / review.py already decode -- avoiding supervision's
        slower, lossy polygon approximation. Each detection's confidence is
        attached inline as ``score`` (bound to its detection at creation, so
        no fragile post-hoc index matching).
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
        (PIL does not decode pixels for ``.size``)."""
        wh = self._image_wh.get(path)
        if wh is not None:
            return wh
        with Image.open(path) as im:
            wh = im.size
        self._image_wh[path] = wh
        return wh

    def _write_coco(
        self, annotations: Dict[str, sv.Detections], progress: bool = True
    ) -> Path:
        """Build COCO from the detections collected so far and write
        annotations.json directly (no supervision DetectionDataset / image
        re-read / double serialization).

        Safe to call repeatedly (checkpointing): only the images already
        processed are included, so a killed run keeps valid partial output.
        Per-image annotation dicts are cached so each mask is RLE-encoded only
        once across the whole run.
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

        items = annotations.items()
        if progress:
            items = tqdm(
                list(items), desc="Writing COCO", leave=False, unit="img"
            )
        for path, det in items:
            anns = self._coco_cache.get(path)
            if anns is None:
                anns = self._detections_to_coco(det)
                self._coco_cache[path] = anns
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
        print(f"Loading Grounding DINO ({self.dino_model_id}) + SAM2 ({self.sam2_model}) on {self.device}...")
        processor, dino, sam = self.load_models()

        image_paths = self._collect_images()
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.source_directory}")
        print(f"Found {len(image_paths)} image(s) in '{self.source_directory}'.")

        checkpoint_every = 25
        annotations: Dict[str, sv.Detections] = {}
        failures: List[str] = []
        for n, path in enumerate(tqdm(image_paths, desc="Segmenting"), start=1):
            try:
                annotations[path] = self.predict_image(path, processor, dino, sam)
            except Exception as e:  # one bad image must not abort the batch
                annotations[path] = sv.Detections.empty()
                failures.append(path)
                tqdm.write(f"FAILED {path}: {type(e).__name__}: {e}")
            if n % checkpoint_every == 0:
                self._write_coco(annotations, progress=False)

        coco_path = self._write_coco(annotations)
        print(f"Done. COCO annotations written to: {coco_path}")
        if failures:
            print(f"\n{len(failures)} image(s) failed and were written empty:")
            for p in failures:
                print(f"  - {p}")
