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

# Optional: automated annotation verification (verify.py). All keys are
# optional and default to the values shown below.
verify:
  model: claude-sonnet-5         # default: claude-sonnet-5
  use_batch: true                # default: true (Batch API, ~50% cheaper; false = one request per frame)
  max_image_width: 1280          # default: 1280 (overlay image sent to the model is downscaled to this width)
  max_tokens: 1024               # default: 1024
```

## Usage

Run the pipeline:

```bash
python run.py
```

This writes `<dest_directory>/annotations.json` in COCO format (categories, images, annotations with `bbox`, `area`, RLE `segmentation`).

To render static visual overlays of the annotations:

```bash
python -m util.render_overlays config.yaml
```

Outputs `overlay_<filename>.jpg` files into `<dest_directory>/overlays/` with boxes, masks, and class labels drawn on each image.

### Automated review

An optional pass that uses Claude to do per-detection quality control (keep / reject / reclassify) before the human review UI, auto-resolving the confident majority of detections and flagging only genuinely uncertain ones for a human yes/no pass:

```bash
export ANTHROPIC_API_KEY=...
python verify.py --config config.yaml
```

- `--limit N`: only verify the first N frames (smoke test).
- `--sync`: use synchronous per-frame requests instead of the Batch API — useful for a quick smoke test without waiting on batch completion (pair with `--limit`).

Verdicts are schema-constrained (structured outputs), so responses are guaranteed valid JSON with a `class` restricted to the actual ontology. At 100k standard-resolution frames, the default model + Batch API path costs roughly $300-450. Cost, model, and image size are configurable via the `verify:` block in `config.yaml` (see Configuration above).

Writes `<dest_directory>/review_state.json` (pre-filled decisions; confidently-resolved frames pre-marked reviewed), `<dest_directory>/reviewed_annotations.json` (confident keeps/reclassifies only), and `<dest_directory>/verification_report.json` (counts, flag reasons, token usage). None of this modifies `annotations.json`. Frames the verifier flagged remain unreviewed with the verifier's tentative decision pre-applied, so opening the Review UI next (below) means only touching that flagged minority.

### Review UI

An interactive Streamlit app for cleaning auto-labels before downstream training:

```bash
streamlit run review.py -- --config config.yaml
```

(`--config` defaults to `config.yaml`.) It opens a browser tab and lets you step through every frame and:

- **Click an object in the image** to select it — its controls pin to the top of the sidebar and its box is highlighted (yellow if kept, orange if rejected).
- **Accept / reject** each detection (kept by default — you only reject the wrong ones) and **reclassify** via the class dropdown. Detections are listed lowest-confidence first.
- **Save** a clean, kept-only `<dest_directory>/reviewed_annotations.json` (standard COCO, per-detection `score` preserved).

Progress autosaves to `<dest_directory>/review_state.json` on navigation, so closing and relaunching resumes exactly where you left off. The source `annotations.json` is never modified.

## Output format

`annotations.json` follows COCO format:
- `categories`: numbered classes from the ontology (values, not prompts)
- `images`: one entry per source image with width/height
- `annotations`: one per detection — `bbox` (xywh), `area`, `segmentation` (compressed RLE mask), `category_id`, and `score` (Grounding DINO detection confidence in `[0, 1]`; `-1.0` if it could not be aligned)

Per-frame post-filter: only the highest-confidence ball detection is kept per image (structural constraint — soccer has one ball in play).

The review UI additionally produces `reviewed_annotations.json` (the cleaned, kept-only subset for training) and `review_state.json` (per-annotation review progress, for resume — not a training artifact). Running `verify.py` first populates both of those (see Automated review above) plus `verification_report.json` (run summary — counts, flag reasons, token usage — not a training artifact).

## Notes

- First run downloads ~1.5 GB of model weights into the HuggingFace cache. Subsequent runs are fast.
- Inference speed: ~2 it/s on an RTX 3090 with `grounding-dino-base` + `sam2-hiera-large`. `grounding-dino-tiny` is ~3x faster but has noticeably lower recall — at the default `box_threshold` it can miss small/numerous classes (e.g. players) entirely. `grounding-dino-base` with `box_threshold: 0.25` is recommended for soccer footage.
- Prompt → class mapping is word-level and tolerant of how Grounding DINO returns labels: multi-word prompts (`"soccer player"`) still resolve when the model returns only a fragment (`"player"`), while labels that are ambiguous across classes (e.g. bare `"soccer"`, shared by player/ball/field) are dropped rather than misassigned.
- This pipeline is intentionally a **labeler**, not a production detector. Output is meant to be reviewed/cleaned before training a custom downstream model.
