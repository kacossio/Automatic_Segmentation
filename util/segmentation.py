"""
Text-prompted image segmentation using HF Grounding DINO + SAM2.
Produces COCO JSON output from per-image masks.
"""
import os
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
        label = label.lower().strip()
        if label in self.prompt_to_class_id:
            return self.prompt_to_class_id[label]
        # Fallback: longest prompt that appears as substring of label
        best: Optional[str] = None
        for p in self.prompts:
            if p in label and (best is None or len(p) > len(best)):
                best = p
        return self.prompt_to_class_id[best] if best else None

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
        print(f"Done. COCO annotations written to: {coco_path}")
