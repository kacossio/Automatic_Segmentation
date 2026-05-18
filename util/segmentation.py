"""
Text-prompted image segmentation using HF Grounding DINO + SAM2.
Produces COCO JSON output from per-image masks.
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import supervision as sv
import torch
import yaml
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class Segmenter:

    def __init__(self, config: str):
        with open(config) as f:
            cfg = yaml.safe_load(f)

        self.source_directory: str = cfg["source_directory"]
        self.dest_directory: str = cfg["dest_directory"]
        self.ontology_map: Dict[str, str] = cfg["ontology"]

        self.dino_model_id: str = cfg.get("dino_model", "IDEA-Research/grounding-dino-tiny")
        self.sam2_model: str = cfg.get("sam2_model", "facebook/sam2-hiera-large")
        self.box_threshold: float = float(cfg.get("box_threshold", 0.35))
        self.text_threshold: float = float(cfg.get("text_threshold", 0.25))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.prompts: List[str] = [p.lower().strip() for p in self.ontology_map.keys()]
        self.classes: List[str] = list(dict.fromkeys(self.ontology_map.values()))
        # Map normalized prompt -> class id
        self.prompt_to_class_id: Dict[str, int] = {
            p.lower().strip(): self.classes.index(c) for p, c in self.ontology_map.items()
        }

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

        keep: List[int] = []
        class_ids: List[int] = []
        for i, label in enumerate(text_labels):
            cid = self._label_to_class_id(label)
            if cid is not None:
                keep.append(i)
                class_ids.append(cid)

        if not keep:
            return sv.Detections.empty()

        boxes = boxes[keep]
        scores = scores[keep]

        image_rgb = np.asarray(pil_image)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            sam.set_image(image_rgb)
            masks_out: List[np.ndarray] = []
            for box in boxes:
                masks, mask_scores, _ = sam.predict(box=box, multimask_output=False)
                masks_out.append(masks[int(np.argmax(mask_scores))].astype(bool))

        det = sv.Detections(
            xyxy=boxes,
            mask=np.stack(masks_out, axis=0),
            class_id=np.array(class_ids, dtype=int),
            confidence=scores,
        )
        return self._apply_domain_filters(det)

    def _apply_domain_filters(self, det: sv.Detections) -> sv.Detections:
        """Soccer-specific post-filter: keep only the highest-confidence ball
        detection per image (only one ball can be in play)."""
        if len(det) == 0:
            return det

        cls_index = {name: i for i, name in enumerate(self.classes)}
        keep = np.ones(len(det), dtype=bool)

        ball_id = cls_index.get("ball")
        if ball_id is not None:
            ball_idx = np.where(det.class_id == ball_id)[0]
            if len(ball_idx) > 1:
                best = ball_idx[int(np.argmax(det.confidence[ball_idx]))]
                keep[ball_idx] = False
                keep[best] = True

        return det[keep]

    def _collect_images(self) -> List[str]:
        root = Path(self.source_directory)
        paths: List[str] = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(str(p) for p in root.glob(f"*{ext}"))
        return sorted(paths)

    def run(self):
        print(f"Loading Grounding DINO ({self.dino_model_id}) + SAM2 ({self.sam2_model}) on {self.device}...")
        processor, dino, sam = self.load_models()

        image_paths = self._collect_images()
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.source_directory}")
        print(f"Found {len(image_paths)} image(s) in '{self.source_directory}'.")

        annotations: Dict[str, sv.Detections] = {}
        for path in tqdm(image_paths, desc="Segmenting"):
            annotations[path] = self.predict_image(path, processor, dino, sam)

        dataset = sv.DetectionDataset(
            classes=self.classes,
            images=image_paths,
            annotations=annotations,
        )

        coco_path = Path(self.dest_directory) / "annotations.json"
        dataset.as_coco(annotations_path=str(coco_path))

        # supervision's as_coco() does not persist Detections.confidence.
        # Re-load the JSON and attach each detection's score as a custom field.
        # as_coco writes annotations grouped per image in detection-index order,
        # which matches the order of each Detections.confidence array.
        with open(coco_path) as f:
            coco = json.load(f)

        scores_by_filename: Dict[str, List[float]] = {
            Path(p).name: (
                det.confidence.tolist()
                if det.confidence is not None and len(det) > 0
                else []
            )
            for p, det in annotations.items()
        }
        id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

        per_image_idx: Dict[int, int] = {}
        for a in coco["annotations"]:
            fn = id_to_filename.get(a["image_id"])
            scores = scores_by_filename.get(fn, [])
            i = per_image_idx.get(a["image_id"], 0)
            if i < len(scores):
                a["score"] = float(scores[i])
            else:
                a["score"] = -1.0
                print(f"WARNING: score count mismatch for '{fn}'; set score=-1.0")
            per_image_idx[a["image_id"]] = i + 1

        with open(coco_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"Done. COCO annotations written to: {coco_path}")
