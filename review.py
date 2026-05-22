"""Streamlit annotation review UI.

Step through the COCO frames produced by Segmenter, inspect every detection
with its confidence, and reject / reclassify the wrong ones. Emits a clean
kept-only ``reviewed_annotations.json`` for downstream training plus a
separate ``review_state.json`` that makes review fully resumable.

Usage:
    streamlit run review.py -- --config config.yaml

``--config`` defaults to config.yaml. Unlike run.py (which hardcodes its
config) this app takes the path explicitly so the same UI can review any
configured output directory.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import supervision as sv
import yaml
from pycocotools import mask as coco_mask
from streamlit_image_coordinates import streamlit_image_coordinates

# NOTE: torch / sam2 are imported lazily inside load_sam(), NOT at module top.
# Streamlit's source watcher walks every imported module's __path__, and
# torch.classes.__path__._path raises (RuntimeError: Tried to instantiate
# class '__path__._path'), which crashes the app on every rerun. Deferring the
# import keeps pure review torch-free; load_sam() also silences the watcher.

from util.render_overlays import coco_anns_to_detections

# Max width (px) of the frame sent to the browser-side click component.
# Hit-testing stays full-resolution; only the displayed copy is shrunk.
DISPLAY_MAX_W = 1280


# --------------------------------------------------------------------------- #
# Config / data loading
# --------------------------------------------------------------------------- #
def parse_args():
    # Streamlit passes script args after "--"; ignore anything it adds.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


@st.cache_data(show_spinner=False)
def load_coco(config_path: str, coco_mtime: float):
    # coco_mtime is unused inside but participates in the cache key: re-running
    # the labeler changes annotations.json's mtime and invalidates this cache,
    # so the UI never shows stale annotations.
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    src = Path(cfg["source_directory"])
    dest = Path(cfg["dest_directory"])
    with open(dest / "annotations.json") as f:
        coco = json.load(f)

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    images = list(coco["images"])  # preserve file order
    per_img: dict[int, list] = {}
    for a in coco["annotations"]:
        per_img.setdefault(a["image_id"], []).append(a)
    return cfg, str(src), str(dest), coco, cats, images, per_img


@st.cache_resource(show_spinner=False)
def load_frame(img_path: str, _anns, _cats, cache_key: str):
    """Read an image and decode all its detections once (cached per frame).

    ``cache_key`` (the image id) is the hash key; ``_anns``/``_cats`` are
    excluded from hashing so a checkbox toggle does not re-decode masks.

    Uses ``cache_resource`` (not ``cache_data``) so a cache hit returns the
    same objects instead of pickling/unpickling the image + stacked masks on
    every rerun. Safe because callers never mutate the cached objects: the
    image is ``.copy()``- d before drawing and ``det`` is sliced into a new
    ``Detections`` before any field is reassigned.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None
    h, w = img.shape[:2]
    if not _anns:
        return img, sv.Detections.empty(), []
    det, labels = coco_anns_to_detections(_anns, h, w, _cats)
    return img, det, labels


@st.cache_resource(show_spinner="Loading SAM2 (first add only)…")
def load_sam(sam2_model: str):
    """Lazily load SAM2 for the click-to-add flow.

    Only invoked the first time the reviewer adds an object, so pure
    review on a CPU-only box never pays the model-load cost — and, crucially,
    torch is not imported until then so Streamlit's file watcher never trips
    over torch.classes.
    """
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Stop Streamlit's source watcher from walking torch.classes.__path__
    # (raises RuntimeError on every rerun once torch is imported).
    try:
        torch.classes.__path__ = []
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SAM2ImagePredictor.from_pretrained(sam2_model, device=device), device


def sam_mask_at_point(sam, bgr_image, ox: int, oy: int) -> np.ndarray:
    """Point-prompt SAM2 at (ox, oy); return the best-scoring boolean mask."""
    import torch  # already loaded by load_sam(); import is cached/cheap

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        sam.set_image(rgb)
        masks, scores, _ = sam.predict(
            point_coords=np.array([[ox, oy]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
    return masks[int(np.argmax(scores))].astype(bool)


def make_added_ann(ann_id: int, image_id: int, mask: np.ndarray, category_id: int):
    """Build a COCO annotation (compressed RLE) for a human-added mask.

    ``score`` is 1.0 and ``category_id`` defaults to the UNKNOWN class so the
    reviewer must explicitly classify it via the existing dropdown.
    """
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    ys, xs = np.where(mask)
    x0, y0 = int(xs.min()), int(ys.min())
    bw, bh = int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x0, y0, bw, bh],
        "area": int(mask.sum()),
        "iscrowd": 0,
        "segmentation": {
            "size": [int(mask.shape[0]), int(mask.shape[1])],
            "counts": rle["counts"].decode("ascii"),
        },
        "score": 1.0,
    }


def all_anns(per_img: dict, img_id: int) -> list:
    """Labeler detections for a frame plus any human-added ones (session)."""
    return per_img.get(img_id, []) + st.session_state.get("added", {}).get(img_id, [])


# --------------------------------------------------------------------------- #
# State persistence (review_state.json)
# --------------------------------------------------------------------------- #
def dkey(image_id: int, ann_id: int) -> str:
    return f"{image_id}:{ann_id}"


def state_path(dest: str) -> Path:
    return Path(dest) / "review_state.json"


def init_state(dest: str, per_img: dict, images: list):
    """Hydrate decisions + reviewed set, resuming from disk if present."""
    decisions: dict[str, dict] = {}
    reviewed: set[int] = set()
    added: dict[int, list] = {}
    next_added_id = -1  # human-added anns use decreasing negative ids

    sp = state_path(dest)
    if sp.exists():
        with open(sp) as f:
            saved = json.load(f)
        decisions = saved.get("decisions", {})
        reviewed = set(saved.get("reviewed", []))
        # JSON object keys are strings; restore int image-id keys.
        added = {int(k): v for k, v in saved.get("added", {}).items()}
        next_added_id = saved.get("next_added_id", -1)

    st.session_state.added = added
    st.session_state.next_added_id = next_added_id

    # Ensure every annotation (incl. resumed human-added ones) has a decision
    # (inclusive default = keep).
    for img in images:
        iid = img["id"]
        for a in per_img.get(iid, []) + added.get(iid, []):
            k = dkey(iid, a["id"])
            if k not in decisions:
                decisions[k] = {"keep": True, "class_id": a["category_id"]}

    st.session_state.decisions = decisions
    st.session_state.reviewed = reviewed
    st.session_state.current_idx = 0
    st.session_state.selected = {}  # image_id -> selected annotation id
    st.session_state.dirty = False  # unsaved decision/reviewed changes


def save_state(dest: str):
    with open(state_path(dest), "w") as f:
        json.dump(
            {
                "decisions": st.session_state.decisions,
                "reviewed": sorted(st.session_state.reviewed),
                "added": st.session_state.get("added", {}),
                "next_added_id": st.session_state.get("next_added_id", -1),
            },
            f,
            indent=2,
        )
    st.session_state.dirty = False


def save_reviewed(dest: str, coco: dict, per_img: dict):
    """Write a clean, kept-only standard-COCO file (score retained)."""
    decisions = st.session_state.decisions
    added = st.session_state.get("added", {})
    out_anns = []
    for img_id in set(per_img) | set(added):
        for a in per_img.get(img_id, []) + added.get(img_id, []):
            d = decisions.get(dkey(img_id, a["id"]))
            if not d or not d["keep"]:
                continue
            na = dict(a)
            na["category_id"] = d["class_id"]
            out_anns.append(na)

    out = {
        "images": coco["images"],
        "categories": coco["categories"],
        "annotations": out_anns,
    }
    for opt in ("info", "licenses"):
        if opt in coco:
            out[opt] = coco[opt]

    path = Path(dest) / "reviewed_annotations.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path, len(out_anns)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def render_row(a, img_id, cats, cat_names, name_to_id, selected=False):
    """Render one detection's keep checkbox + class dropdown.

    Widget keys are stable per (image, annotation) so state survives reruns
    and navigation. Must be called exactly once per annotation per run
    (duplicate keys raise) -- the selected row is pinned at the top and
    skipped in the list below.
    """
    k = dkey(img_id, a["id"])
    d = st.session_state.decisions[k]
    score = float(a.get("score", 1.0))
    kkey, ckey = f"keep::{k}", f"cls::{k}"
    if kkey not in st.session_state:
        st.session_state[kkey] = d["keep"]
    if ckey not in st.session_state:
        st.session_state[ckey] = cats[d["class_id"]]

    label = f"{score:.2f}" if score >= 0 else "⚠"
    if selected:
        label = f"▶ {label}"
    r1, r2 = st.columns([1, 3])
    r1.checkbox(label, key=kkey, help="Keep this detection")
    r2.selectbox("class", cat_names, key=ckey, label_visibility="collapsed")
    new_keep = st.session_state[kkey]
    new_cls = name_to_id[st.session_state[ckey]]
    if d["keep"] != new_keep or d["class_id"] != new_cls:
        st.session_state.dirty = True
    d["keep"] = new_keep
    d["class_id"] = new_cls


def hit_test(det, kept_mask, ox, oy):
    """Return the index of the kept detection at point (ox, oy).

    Prefers the smallest-area mask containing the point (so a player wins
    over the ~full-frame field mask); falls back to the smallest bbox.
    """
    best, best_area = None, None
    H, W = det.mask.shape[1:] if det.mask is not None else (0, 0)
    for i in range(len(det)):
        if not kept_mask[i]:
            continue
        inside = False
        if det.mask is not None and 0 <= oy < H and 0 <= ox < W:
            inside = bool(det.mask[i, oy, ox])
        if not inside:
            x1, y1, x2, y2 = det.xyxy[i]
            inside = x1 <= ox <= x2 and y1 <= oy <= y2
        if inside:
            x1, y1, x2, y2 = det.xyxy[i]
            area = max(1.0, (x2 - x1) * (y2 - y1))
            if best_area is None or area < best_area:
                best, best_area = i, area
    return best


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Annotation Review", layout="wide")
    args = parse_args()

    # Cheap re-read of the config to locate annotations.json and stamp its
    # mtime into the cache key (see load_coco).
    with open(args.config) as f:
        _dest = Path(yaml.safe_load(f)["dest_directory"])
    coco_mtime = os.path.getmtime(_dest / "annotations.json")
    cfg, src, dest, coco, cats, images, per_img = load_coco(args.config, coco_mtime)
    cat_names = [cats[cid] for cid in sorted(cats)]
    name_to_id = {cats[cid]: cid for cid in cats}
    sam2_model = cfg.get("sam2_model", "facebook/sam2-hiera-large")
    # Human-added detections default to UNKNOWN so the reviewer must classify.
    unknown_cid = name_to_id.get("unknown", sorted(cats)[0])

    if "decisions" not in st.session_state:
        init_state(dest, per_img, images)
    st.session_state.setdefault("selected", {})

    n = len(images)
    idx = st.session_state.current_idx
    img_meta = images[idx]
    img_id = img_meta["id"]
    anns = all_anns(per_img, img_id)

    # ---- Top bar -------------------------------------------------------- #
    c1, c2, c3, c4 = st.columns([3, 1, 1, 3])
    done = "✅ reviewed" if img_id in st.session_state.reviewed else "⬜ unreviewed"
    c1.markdown(f"**{img_meta['file_name']}** &nbsp; ({done})")
    c2.markdown(f"image **{idx + 1} / {n}**")

    def go(new_idx: int):
        # Navigation no longer auto-marks a frame reviewed: "reviewed" must be
        # an explicit action so progress is honest against a 100% bar.
        st.session_state.current_idx = max(0, min(n - 1, new_idx))
        if st.session_state.dirty:  # skip the full-state write when nothing changed
            save_state(dest)
        st.rerun()

    if c3.button("◀ Prev", disabled=idx == 0, width="stretch"):
        go(idx - 1)
    if c3.button("Next ▶", disabled=idx == n - 1, width="stretch"):
        go(idx + 1)
    if c3.button(
        "✔ Mark reviewed",
        width="stretch",
        disabled=img_id in st.session_state.reviewed,
    ):
        st.session_state.reviewed.add(img_id)
        st.session_state.dirty = True
        save_state(dest)
        st.rerun()
    jump = c4.selectbox(
        "Jump to",
        range(n),
        index=idx,
        format_func=lambda i: f"{i + 1}. {images[i]['file_name']}",
        label_visibility="collapsed",
    )
    if jump != idx:
        go(jump)

    img_path = str(Path(src) / img_meta["file_name"])
    # Cache key includes the detection count so a freshly added mask busts the
    # per-frame cache and is decoded/shown on the next run.
    image, det, labels = load_frame(
        img_path, anns, cats, f"{img_id}:{len(anns)}"
    )

    main_col, side_col = st.columns([3, 1])
    order = sorted(range(len(anns)), key=lambda i: float(anns[i].get("score", 1.0)))

    # Sync decisions from committed widget state so the overlay reflects the
    # latest checkbox/dropdown values on this run (rows are redrawn below).
    for a in anns:
        k = dkey(img_id, a["id"])
        dec = st.session_state.decisions[k]
        if f"keep::{k}" in st.session_state:
            new_keep = st.session_state[f"keep::{k}"]
            if dec["keep"] != new_keep:
                dec["keep"] = new_keep
                st.session_state.dirty = True
        if f"cls::{k}" in st.session_state:
            new_cls = name_to_id[st.session_state[f"cls::{k}"]]
            if dec["class_id"] != new_cls:
                dec["class_id"] = new_cls
                st.session_state.dirty = True

    kept_mask = [st.session_state.decisions[dkey(img_id, a["id"])]["keep"] for a in anns]
    override_ids = [
        st.session_state.decisions[dkey(img_id, a["id"])]["class_id"] for a in anns
    ]
    sel_id = st.session_state.selected.get(img_id)
    sel_pos = next((i for i, a in enumerate(anns) if a["id"] == sel_id), None)

    # ---- Main: clickable overlay ---------------------------------------- #
    with main_col:
        if image is None:
            st.error(f"Image not found: {img_path}")
        else:
            mode = st.radio(
                "Click mode",
                ["Select", "Add object"],
                horizontal=True,
                key="click_mode",
                help="Select: inspect a detection. Add object: click a missed "
                "object — SAM2 generates a mask the labeler never produced.",
            )

            out = image.copy()
            keep_arr = np.array(kept_mask, dtype=bool)
            # Overlay only when there are kept detections; an empty frame still
            # renders the raw image so a missed-everything frame is addable.
            if len(det) and keep_arr.any():
                shown = det[keep_arr]
                ids = np.array(override_ids, dtype=int)[keep_arr]
                shown.class_id = ids
                shown_labels = [cats[c] for c in ids]
                out = sv.MaskAnnotator(opacity=0.4).annotate(out, shown)
                out = sv.BoxAnnotator().annotate(out, shown)
                out = sv.LabelAnnotator(
                    text_scale=0.5, text_thickness=1
                ).annotate(out, shown, labels=shown_labels)
            # Emphasize the selected detection on top (yellow if kept, else
            # orange so a rejected-but-selected box is still locatable).
            if sel_pos is not None:
                x1, y1, x2, y2 = (int(v) for v in det.xyxy[sel_pos])
                col = (0, 255, 255) if kept_mask[sel_pos] else (0, 165, 255)
                cv2.rectangle(out, (x1, y1), (x2, y2), col, 4)

            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            # Ship a downscaled copy to the browser: the component base64-
            # encodes the array into the page on every rerun, so a smaller
            # array = far less transfer/decode. Click coords are normalized
            # by the reported render width/height below, so this is
            # transparent to hit-testing regardless of display size.
            if rgb.shape[1] > DISPLAY_MAX_W:
                scale = DISPLAY_MAX_W / rgb.shape[1]
                rgb = cv2.resize(
                    rgb,
                    (DISPLAY_MAX_W, int(round(rgb.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            click = streamlit_image_coordinates(
                rgb, key=f"img::{img_id}", use_column_width="always"
            )
            st.caption(
                "Select: click an object to inspect it. "
                "Add object: click a missed object — SAM2 makes a mask "
                "(defaults to “unknown”; classify it in the sidebar)."
            )
            # The component re-emits the last click on every rerun; only act
            # on a *new* click so an add isn't repeated indefinitely.
            last_key = f"_click::{img_id}"
            if (
                click is not None
                and click.get("width")
                and click != st.session_state.get(last_key)
            ):
                st.session_state[last_key] = click
                ox = int(click["x"] * image.shape[1] / click["width"])
                oy = int(click["y"] * image.shape[0] / click["height"])
                if mode == "Add object":
                    sam, _device = load_sam(sam2_model)
                    mask = sam_mask_at_point(sam, image, ox, oy)
                    if not mask.any():
                        # An all-false mask would crash make_added_ann's bbox
                        # (xs.min() on an empty array); skip and tell the user.
                        st.warning(
                            "SAM2 returned an empty mask there — try clicking "
                            "nearer the object's centre."
                        )
                    else:
                        aid = st.session_state.next_added_id
                        st.session_state.next_added_id -= 1
                        st.session_state.added.setdefault(img_id, []).append(
                            make_added_ann(aid, img_id, mask, unknown_cid)
                        )
                        st.session_state.decisions[dkey(img_id, aid)] = {
                            "keep": True,
                            "class_id": unknown_cid,
                        }
                        st.session_state.selected[img_id] = aid
                        st.session_state.dirty = True
                        save_state(dest)
                        st.rerun()
                else:
                    hit = hit_test(det, kept_mask, ox, oy)
                    new_id = anns[hit]["id"] if hit is not None else sel_id
                    if new_id != sel_id:
                        st.session_state.selected[img_id] = new_id
                        st.rerun()

    # ---- Sidebar: selected pinned, then ascending-confidence list -------- #
    with side_col:
        st.markdown("#### Detections")
        if sel_pos is not None:
            st.markdown("**Selected**")
            render_row(
                anns[sel_pos], img_id, cats, cat_names, name_to_id, selected=True
            )
            st.divider()
        for pos in order:
            a = anns[pos]
            if a["id"] == sel_id:
                continue
            render_row(a, img_id, cats, cat_names, name_to_id)

    # ---- Footer --------------------------------------------------------- #
    kept = sum(1 for v in st.session_state.decisions.values() if v["keep"])
    total = len(st.session_state.decisions)
    f1, f2 = st.columns([1, 3])
    if f1.button("💾 Save reviewed.json", type="primary"):
        st.session_state.reviewed.add(img_id)
        save_state(dest)
        path, kept_n = save_reviewed(dest, coco, per_img)
        st.success(f"Wrote {kept_n} kept annotations to {path}")
    f2.markdown(f"**{kept} / {total}** annotations kept across all frames")

    # Optional keyboard shortcuts; silently degrade to buttons if the package
    # isn't installed. Keys map to button labels rendered above.
    try:
        from streamlit_shortcuts import add_keyboard_shortcuts

        add_keyboard_shortcuts(
            {
                "ArrowLeft": "◀ Prev",
                "ArrowRight": "Next ▶",
                "r": "✔ Mark reviewed",
                "s": "💾 Save reviewed.json",
            }
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
