import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.pose_detector import MediaPipePoseMovementAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe pose movement analysis on a saved video."
    )
    parser.add_argument("video_path", type=Path, help="Path to the video to inspect.")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=24,
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

    analyzer = MediaPipePoseMovementAnalyzer()
    summary = analyzer.analyze_video(
        video_path=video_path,
        sample_count=args.sample_count,
    )

    print(f"Video: {video_path}")
    print(f"Frames sampled: {summary.frames_sampled}")
    print(f"Pose frames: {summary.pose_frames}")
    print(f"Pose coverage: {summary.pose_coverage:.2f}")
    print(f"Mean motion: {summary.mean_motion:.4f}")
    print(f"Max motion: {summary.max_motion:.4f}")
    print(f"Vertical motion: {summary.vertical_motion:.4f}")
    print(f"Horizontal motion: {summary.horizontal_motion:.4f}")
    print(f"Posture change: {summary.posture_change:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
