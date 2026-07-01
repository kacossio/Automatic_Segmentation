import argparse
import json

from util.verify import Verifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-verify annotations with Claude, flagging uncertain "
        "detections for human review in review.py."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML config (default: config.yaml).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only verify the first N frames (smoke test).",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous per-frame requests instead of the Batch API.",
    )
    args = parser.parse_args()

    verifier = Verifier(config=args.config)
    report = verifier.run(limit=args.limit, sync=args.sync)
    print(json.dumps(report, indent=2))
