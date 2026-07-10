import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.inference.video_action_classifier import build_r3d18_model
from ml.training.dataset_manifest import load_training_manifest
from ml.training.video_action_dataset import (
    VideoActionDataset,
    VideoActionSample,
    build_label_map,
    build_samples,
)
from ml.training.label_groups import grouped_label


DEFAULT_MANIFEST = Path("dataset/training_manifest.jsonl")
DEFAULT_OUTPUT = Path("ml/models/video_action_classifier.pt")
DEFAULT_LABEL_MAP = Path("ml/models/video_action_labels.json")
DEFAULT_GROUPED_OUTPUT = Path("ml/models/video_action_grouped_classifier.pt")
DEFAULT_GROUPED_LABEL_MAP = Path("ml/models/video_action_grouped_labels.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained TorchVision video action classifier."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--label-map-output", type=Path, default=None)
    parser.add_argument(
        "--label-mode",
        choices=("detailed", "grouped"),
        default="detailed",
        help="Train on detailed behaviour labels or broader grouped risk labels.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=112)
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--test-every", type=int, default=5)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load pretrained Kinetics weights.",
    )
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Train the whole model instead of only the final classification head.",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = (
            DEFAULT_GROUPED_OUTPUT
            if args.label_mode == "grouped"
            else DEFAULT_OUTPUT
        )
    if args.label_map_output is None:
        args.label_map_output = (
            DEFAULT_GROUPED_LABEL_MAP
            if args.label_mode == "grouped"
            else DEFAULT_LABEL_MAP
        )

    items = load_training_manifest(args.manifest)
    if not items:
        raise ValueError(f"No training items found in {args.manifest}")

    label_transform = grouped_label if args.label_mode == "grouped" else None
    label_map = build_label_map(items, label_transform=label_transform)
    samples = build_samples(
        items=items,
        label_map=label_map,
        max_per_label=args.max_per_label,
        label_transform=label_transform,
    )
    train_samples, test_samples = split_samples(samples, test_every=args.test_every)
    if not train_samples or not test_samples:
        raise ValueError("Both train and test samples are required")

    labels = labels_from_map(label_map)
    train_dataset = VideoActionDataset(
        train_samples,
        clip_frames=args.clip_frames,
        frame_size=args.frame_size,
    )
    test_dataset = VideoActionDataset(
        test_samples,
        clip_frames=args.clip_frames,
        frame_size=args.frame_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_r3d18_model(
        num_classes=len(labels),
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )
        accuracy = evaluate(model=model, loader=test_loader, device=device)
        best_accuracy = max(best_accuracy, accuracy)
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"test_accuracy={accuracy:.3f}"
        )

    save_checkpoint(
        output_path=args.output,
        label_map_output=args.label_map_output,
        model=model,
        labels=labels,
        label_map=label_map,
        train_count=len(train_samples),
        test_count=len(test_samples),
        best_accuracy=best_accuracy,
        clip_frames=args.clip_frames,
        frame_size=args.frame_size,
        label_mode=args.label_mode,
    )
    print(f"Wrote video action classifier checkpoint to {args.output}")


def split_samples(
    samples: list[VideoActionSample],
    test_every: int,
) -> tuple[list[VideoActionSample], list[VideoActionSample]]:
    if test_every < 2:
        raise ValueError("test_every must be at least 2")

    by_label: dict[str, list[VideoActionSample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)

    train_samples = []
    test_samples = []
    for label_samples in by_label.values():
        shuffled = list(label_samples)
        random.Random(42).shuffle(shuffled)
        for index, sample in enumerate(shuffled, start=1):
            if len(shuffled) > 1 and index % test_every == 0:
                test_samples.append(sample)
            else:
                train_samples.append(sample)

    return train_samples, test_samples


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for videos, labels in loader:
        videos = videos.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(videos)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())

    return total_loss / max(1, len(loader))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.to(device)
            predictions = model(videos).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return correct / total if total else 0.0


def save_checkpoint(
    output_path: Path,
    label_map_output: Path,
    model: nn.Module,
    labels: list[str],
    label_map: dict[str, int],
    train_count: int,
    test_count: int,
    best_accuracy: float,
    clip_frames: int,
    frame_size: int,
    label_mode: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "torchvision_r3d18",
            "label_mode": label_mode,
            "model_state_dict": model.state_dict(),
            "labels": labels,
            "train_count": train_count,
            "test_count": test_count,
            "best_accuracy": best_accuracy,
            "clip_frames": clip_frames,
            "frame_size": frame_size,
        },
        output_path,
    )

    label_map_output.parent.mkdir(parents=True, exist_ok=True)
    with label_map_output.open("w", encoding="utf-8") as output_file:
        json.dump(label_map, output_file, indent=2)


def labels_from_map(label_map: dict[str, int]) -> list[str]:
    return [
        label
        for label, _ in sorted(label_map.items(), key=lambda item: item[1])
    ]


if __name__ == "__main__":
    main()
