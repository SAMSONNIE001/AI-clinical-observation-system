import argparse
from pathlib import Path

import torch


DEFAULT_CHECKPOINT = Path("ml/models/video_action_grouped_classifier.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a saved video action classifier checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path to inspect.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return 1

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {checkpoint.get('model_type')}")
    print(f"Label mode: {checkpoint.get('label_mode', 'detailed')}")
    print(f"Labels: {checkpoint.get('labels')}")
    print(f"Train count: {checkpoint.get('train_count')}")
    print(f"Test count: {checkpoint.get('test_count')}")
    print(f"Best epoch: {checkpoint.get('best_epoch')}")
    print(f"Best accuracy: {checkpoint.get('best_accuracy')}")

    history = checkpoint.get("history") or []
    if history:
        print("History:")
        for row in history:
            print(
                f"  epoch={row['epoch']} "
                f"train_loss={row['train_loss']:.4f} "
                f"test_accuracy={row['test_accuracy']:.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
