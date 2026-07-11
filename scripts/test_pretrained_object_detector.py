import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.object_detector import (
    DEFAULT_YOLO_MODEL_PATH,
    PretrainedObjectDetector,
    dangerous_objects_from_detections,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pretrained YOLO object detector on a saved video."
    )
    parser.add_argument("video_path", type=Path, help="Path to the video to inspect.")
    parser.add_argument(
        "--model",
        default=DEFAULT_YOLO_MODEL_PATH,
        help="YOLO model name or local model path.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum object detection confidence.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=12,
        help="Approximate number of frames to sample from the video.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = args.video_path
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return 1

    detector = PretrainedObjectDetector(
        model_name=args.model,
        confidence_threshold=args.confidence,
    )
    detections = detector.detect_video(
        video_path=video_path,
        sample_count=args.sample_count,
    )
    dangerous_objects = dangerous_objects_from_detections(detections)

    print(f"Video: {video_path}")
    print(f"Detections: {len(detections)}")
    print(f"Dangerous objects: {dangerous_objects}")

    for detection in detections:
        print(
            f"frame={detection.frame_index} "
            f"label={detection.label} "
            f"confidence={detection.confidence:.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
