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

    Decodes COCO segmentation back into boolean masks and rehydrates an
    sv.Detections. ``confidence`` is taken from each annotation's optional
    ``score`` field (default 1.0). ``cats`` maps category_id -> class name.
    Shared by render_overlays() and review.py.

    Handles both segmentation encodings: polygons / uncompressed RLE (the
    labeler's supervision output, via ``frPyObjects``) and compressed RLE
    (masks added in review.py, where ``counts`` is a str/bytes blob that
    ``frPyObjects`` cannot parse and must be decoded directly).
    """
    xyxy, masks, class_ids, labels, conf = [], [], [], [], []
    for a in anns:
        x, y, bw, bh = a["bbox"]
        xyxy.append([x, y, x + bw, y + bh])
        seg = a["segmentation"]
        if isinstance(seg, dict) and isinstance(seg.get("counts"), (str, bytes)):
            rle = dict(seg)
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("ascii")
            m = coco_mask.decode(rle)
        else:
            m = coco_mask.decode(coco_mask.frPyObjects(seg, h, w))
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


FIELD_OUTLINE_COLOR = (0, 255, 0)  # BGR green
FIELD_OUTLINE_THICKNESS = 2


UNLABELED_CLASSES = ("player",)


def annotate_detections(
    image,
    det,
    labels,
    cats,
    field_class_name: str = "field",
    show_boxes: bool = False,
    unlabeled_classes=UNLABELED_CLASSES,
):
    """Mask overlay for non-field detections; labels only for rare classes;
    outline-only for the field. Bounding boxes are off by default — masks
    already carry the spatial info, and stacked boxes drown the frame in
    noise. Pass ``show_boxes=True`` for a thin debug outline.

    Labels for the dominant class (players) are suppressed because repeating
    the same word on every detection in a crowded scene buries the rare
    classes that *do* need a label (ball, referee, goalkeeper). Pass
    ``unlabeled_classes=()`` to label every detection.

    Why: SAM3's "soccer field" mask spans nearly the whole bottom of the frame
    and its axis-aligned bbox is even larger, so the default filled-mask + box
    + label overlay drowns out every player on top of it. Drawing the field
    as a thin contour preserves the segmentation evidence without obscuring
    anything. Shared by render_overlays() and review.py so the static PNGs
    and the live UI stay consistent.
    """
    out = image.copy()
    if len(det) == 0:
        return out

    is_field = np.array(
        [cats.get(int(c)) == field_class_name for c in det.class_id], dtype=bool
    )

    if (~is_field).any():
        non_field = det[~is_field]
        non_field_labels = [
            lbl for lbl, f in zip(labels, is_field) if not f
        ]
        out = sv.MaskAnnotator(opacity=0.4).annotate(out, non_field)
        if show_boxes:
            out = sv.BoxAnnotator(thickness=1).annotate(out, non_field)

        label_mask = np.array(
            [lbl not in unlabeled_classes for lbl in non_field_labels], dtype=bool
        )
        if label_mask.any():
            labeled = non_field[label_mask]
            labeled_text = [
                lbl for lbl, keep in zip(non_field_labels, label_mask) if keep
            ]
            label_layer = sv.LabelAnnotator(
                text_scale=0.5,
                text_thickness=1,
                text_position=sv.Position.TOP_CENTER,
            ).annotate(out.copy(), labeled, labels=labeled_text)
            # supervision's LabelAnnotator has no opacity arg; blend the
            # rendered label layer back over the base. Untouched pixels
            # blend to themselves (identity), so only the label rectangles
            # become translucent.
            out = cv2.addWeighted(label_layer, 0.7, out, 0.3, 0)

    if is_field.any() and det.mask is not None:
        for i in np.where(is_field)[0]:
            m = det.mask[i].astype(np.uint8)
            contours, _ = cv2.findContours(
                m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(
                out, contours, -1, FIELD_OUTLINE_COLOR, FIELD_OUTLINE_THICKNESS
            )

    return out


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

        out = annotate_detections(img, det, labels, cats)
        cv2.imwrite(str(out_path), out)
        print(f"  {info['file_name']}: {len(anns)} annotations -> {out_path}")

    return out_dir


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    render_overlays(cfg)
