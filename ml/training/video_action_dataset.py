import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ml.training.dataset_manifest import TrainingManifestItem


KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


@dataclass(frozen=True)
class VideoActionSample:
    video_id: str
    video_path: Path
    label: str
    label_index: int


class VideoActionDataset(Dataset):
    def __init__(
        self,
        samples: list[VideoActionSample],
        clip_frames: int = 16,
        frame_size: int = 112,
        training: bool = False,
    ) -> None:
        self.samples = samples
        self.clip_frames = clip_frames
        self.frame_size = frame_size
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        video = load_video_clip(
            video_path=sample.video_path,
            clip_frames=self.clip_frames,
            frame_size=self.frame_size,
            random_sampling=self.training,
            augment=self.training,
        )
        label = torch.tensor(sample.label_index, dtype=torch.long)
        return video, label


def build_label_map(
    items: list[TrainingManifestItem],
    label_transform=None,
) -> dict[str, int]:
    transform = label_transform or (lambda label: label)
    labels = {transform(item.label) for item in items}
    return {label: index for index, label in enumerate(sorted(labels))}


def build_samples(
    items: list[TrainingManifestItem],
    label_map: dict[str, int],
    project_root: Path = Path("."),
    max_per_label: int | None = None,
    label_transform=None,
) -> list[VideoActionSample]:
    transform = label_transform or (lambda label: label)
    counts: dict[str, int] = {}
    samples: list[VideoActionSample] = []

    for item in items:
        label = transform(item.label)
        if item.relative_path is None:
            continue
        if max_per_label is not None and counts.get(label, 0) >= max_per_label:
            continue

        video_path = Path(item.relative_path)
        if not video_path.is_absolute():
            video_path = project_root / video_path
        if not video_path.exists():
            continue

        samples.append(
            VideoActionSample(
                video_id=item.video_id,
                video_path=video_path,
                label=label,
                label_index=label_map[label],
            )
        )
        counts[label] = counts.get(label, 0) + 1

    return samples


def load_video_clip(
    video_path: Path,
    clip_frames: int = 16,
    frame_size: int = 112,
    random_sampling: bool = False,
    augment: bool = False,
) -> torch.Tensor:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"Could not read frame count: {video_path}")

    if random_sampling and frame_count > clip_frames:
        window_size = random.randint(clip_frames, frame_count)
        start_index = random.randint(0, frame_count - window_size)
        stop_index = start_index + window_size - 1
        target_indices = set(
            np.linspace(start_index, stop_index, num=clip_frames, dtype=int)
        )
    else:
        target_indices = set(np.linspace(0, frame_count - 1, num=clip_frames, dtype=int))
    frames: list[np.ndarray] = []
    flip_horizontal = augment and random.random() < 0.5
    brightness_scale = random.uniform(0.85, 1.15) if augment else 1.0

    try:
        frame_index = 0
        while len(frames) < clip_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in target_indices:
                resized = cv2.resize(frame, (frame_size, frame_size))
                if flip_horizontal:
                    resized = cv2.flip(resized, 1)
                if brightness_scale != 1.0:
                    resized = np.clip(
                        resized.astype(np.float32) * brightness_scale,
                        0,
                        255,
                    ).astype(np.uint8)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"Could not sample frames from video: {video_path}")

    while len(frames) < clip_frames:
        frames.append(frames[-1].copy())

    array = np.stack(frames).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2)
    return (tensor - KINETICS_MEAN) / KINETICS_STD


def save_label_map(label_map: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(label_map, output_file, indent=2)
