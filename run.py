import argparse

import util.segmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-label images with Grounding DINO + SAM2."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML config (default: config.yaml).",
    )
    args = parser.parse_args()

    segmenter = util.segmentation.Segmenter(config=args.config)
    segmenter.run()
