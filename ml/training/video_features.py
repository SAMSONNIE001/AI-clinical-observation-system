from pathlib import Path

import cv2
import numpy as np


FEATURE_VERSION = "opencv-frame-stats-v2"


def extract_video_features(
    video_path: Path,
    sample_count: int = 16,
    frame_size: tuple[int, int] = (64, 64),
) -> list[float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0 or fps <= 0:
        capture.release()
        raise ValueError(f"Could not read video metadata: {video_path}")

    target_indices = set(np.linspace(0, frame_count - 1, num=sample_count, dtype=int))
    gray_frames: list[np.ndarray] = []
    colour_means: list[np.ndarray] = []

    frame_index = 0
    while len(gray_frames) < sample_count:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index not in target_indices:
            frame_index += 1
            continue

        resized = cv2.resize(frame, frame_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray_frames.append(gray)
        colour_means.append(resized.mean(axis=(0, 1)) / 255.0)
        frame_index += 1

    capture.release()

    if not gray_frames:
        raise ValueError(f"Could not sample frames from video: {video_path}")

    frames = np.stack(gray_frames)
    colour = np.stack(colour_means)
    duration_seconds = frame_count / fps

    motion = np.abs(np.diff(frames, axis=0)) if len(frames) > 1 else np.zeros_like(frames)
    edges = np.stack([cv2.Canny((frame * 255).astype(np.uint8), 50, 150) for frame in frames])

    feature_vector = np.array(
        [
            duration_seconds / 120.0,
            fps / 120.0,
            frames.mean(),
            frames.std(),
            frames.min(),
            frames.max(),
            motion.mean(),
            motion.std(),
            motion.max(),
            edges.mean() / 255.0,
            edges.std() / 255.0,
            colour[:, 0].mean(),
            colour[:, 1].mean(),
            colour[:, 2].mean(),
            colour[:, 0].std(),
            colour[:, 1].std(),
            colour[:, 2].std(),
        ],
        dtype=np.float32,
    )

    return feature_vector.tolist()
