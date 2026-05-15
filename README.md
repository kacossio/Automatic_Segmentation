# Automatic_Segmentation

Text-prompted image segmentation pipeline using HuggingFace Grounding DINO + Meta SAM2. Walks a folder of images, runs zero-shot detection from a user-defined ontology, generates instance masks with SAM2, and writes everything to COCO JSON suitable for downstream training.

Built for soccer footage auto-labeling (`player`, `goalkeeper`, `referee`, `ball`, `field`), but the ontology is fully configurable for any domain.

## Installation

Requires Python 3.11+ and an NVIDIA GPU with CUDA (tested on CUDA 12.4 / RTX 3090).

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate

# Install PyTorch with CUDA *first* — pinning the index avoids pulling a CPU-only build.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Then the rest:
pip install -r requirements.txt
```

Model weights download automatically on first run to `~/.cache/huggingface/` — no manual download step.

## Configuration

Create a `config.yaml`:

```yaml
source_directory: data                  # folder of input .jpg/.jpeg/.png images (non-recursive)
dest_directory: data/output             # where annotations.json is written

# Optional model + threshold overrides
dino_model: IDEA-Research/grounding-dino-base   # default: grounding-dino-tiny
sam2_model: facebook/sam2-hiera-large           # default: same
box_threshold: 0.25                              # default: 0.35
text_threshold: 0.20                             # default: 0.25

# Map text prompts -> class labels (the COCO category names)
ontology:
  "soccer player": "player"
  "goalkeeper": "goalkeeper"
  "referee": "referee"
  "soccer ball": "ball"
  "soccer field": "field"
```

## Usage

Run the pipeline:

```bash
python run.py
```

This writes `<dest_directory>/annotations.json` in COCO format (categories, images, annotations with `bbox`, `area`, RLE `segmentation`).

To render visual overlays of the annotations for review:

```bash
python -m util.render_overlays config.yaml
```

Outputs `overlay_<filename>.jpg` files into `<dest_directory>/overlays/` with boxes, masks, and class labels drawn on each image.

## Output format

`annotations.json` follows COCO format:
- `categories`: numbered classes from the ontology (values, not prompts)
- `images`: one entry per source image with width/height
- `annotations`: one per detection — `bbox` (xywh), `area`, `segmentation` (compressed RLE mask), `category_id`

Per-frame post-filter: only the highest-confidence ball detection is kept per image (structural constraint — soccer has one ball in play).

## Notes

- First run downloads ~1.5 GB of model weights into the HuggingFace cache. Subsequent runs are fast.
- Inference speed: ~2 it/s on an RTX 3090 with `grounding-dino-base` + `sam2-hiera-large`. Use `grounding-dino-tiny` for ~3x speedup at lower recall.
- This pipeline is intentionally a **labeler**, not a production detector. Output is meant to be reviewed/cleaned before training a custom downstream model.
