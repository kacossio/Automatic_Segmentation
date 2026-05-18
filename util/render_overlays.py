"""Render annotation overlays for visual review.

Reads the COCO output produced by Segmenter and writes
per-image overlay PNGs (boxes + masks + class labels) into
<dest_directory>/overlays/.

Usage:
    python -m util.render_overlays [config.yaml]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml
from pycocotools import mask as coco_mask


def coco_anns_to_detections(anns, h, w, cats):
    """COCO annotation dicts -> (sv.Detections, labels list).

    Decodes COCO RLE segmentation back into boolean masks and rehydrates an
    sv.Detections. ``confidence`` is taken from each annotation's optional
    ``score`` field (default 1.0). ``cats`` maps category_id -> class name.
    Shared by render_overlays() and review.py.
    """
    xyxy, masks, class_ids, labels, conf = [], [], [], [], []
    for a in anns:
        x, y, bw, bh = a["bbox"]
        xyxy.append([x, y, x + bw, y + bh])
        rle = coco_mask.frPyObjects(a["segmentation"], h, w)
        m = coco_mask.decode(rle)
        if m.ndim == 3:
            m = m[:, :, 0]
        masks.append(m.astype(bool))
        class_ids.append(a["category_id"])
        labels.append(cats[a["category_id"]])
        conf.append(float(a.get("score", 1.0)))

    det = sv.Detections(
        xyxy=np.array(xyxy, dtype=float),
        mask=np.stack(masks),
        class_id=np.array(class_ids, dtype=int),
        confidence=np.array(conf, dtype=float),
    )
    return det, labels


def render_overlays(config_path: str) -> Path:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    src = Path(cfg["source_directory"])
    dest = Path(cfg["dest_directory"])
    coco_path = dest / "annotations.json"
    out_dir = dest / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_path) as f:
        coco = json.load(f)

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i for i in coco["images"]}

    per_img = {}
    for a in coco["annotations"]:
        per_img.setdefault(a["image_id"], []).append(a)

    mask_ann = sv.MaskAnnotator(opacity=0.4)
    box_ann = sv.BoxAnnotator()
    label_ann = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    for img_id, info in images.items():
        img = cv2.imread(str(src / info["file_name"]))
        if img is None:
            print(f"  {info['file_name']}: SKIP (image not found at {src})")
            continue
        h, w = img.shape[:2]
        anns = per_img.get(img_id, [])
        out_path = out_dir / f"overlay_{info['file_name']}"

        if not anns:
            cv2.imwrite(str(out_path), img)
            print(f"  {info['file_name']}: 0 annotations")
            continue

        det, labels = coco_anns_to_detections(anns, h, w, cats)

        out = img.copy()
        out = mask_ann.annotate(out, det)
        out = box_ann.annotate(out, det)
        out = label_ann.annotate(out, det, labels=labels)
        cv2.imwrite(str(out_path), out)
        print(f"  {info['file_name']}: {len(anns)} annotations -> {out_path}")

    return out_dir


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    render_overlays(cfg)
