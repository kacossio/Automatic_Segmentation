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
from typing import Dict, List, Tuple

import numpy as np
import supervision as sv
import torch
import yaml
from PIL import Image
from pycocotools import mask as coco_mask
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


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
        # Per-class confidence floor applied on top of sam3_score_threshold.
        # Classes absent here fall back to sam3_score_threshold.
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
        # Each mask is RLE-encoded exactly once; checkpoint writes reuse this
        # instead of re-encoding every image processed so far (was O(N^2)).
        self._coco_cache: Dict[str, List[dict]] = {}

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
        keep = np.ones(len(det), dtype=bool)

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
        print(f"Loading SAM 3 ({self.sam3_model_id}) on {self.device}...")
        processor, model = self.load_models()

        image_paths = self._collect_images()
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.source_directory}")
        print(f"Found {len(image_paths)} image(s) in '{self.source_directory}'.")

        checkpoint_every = 25
        annotations: Dict[str, sv.Detections] = {}
        failures: List[str] = []
        for n, path in enumerate(tqdm(image_paths, desc="Segmenting"), start=1):
            try:
                annotations[path] = self.predict_image(path, processor, model)
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
