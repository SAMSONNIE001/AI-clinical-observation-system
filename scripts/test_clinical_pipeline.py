import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = PROJECT_ROOT / "backend"
for path in (PROJECT_ROOT, BACKEND_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ml.inference.clinical_pipeline import PretrainedClinicalObservationPipeline


MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "baseline_video_classifier.json"
ACTION_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "video_action_grouped_classifier.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the combined clinical observation pipeline on a saved video."
    )
    parser.add_argument("video_path", type=Path, help="Path to the video to inspect.")
    parser.add_argument(
        "--disable-objects",
        action="store_true",
        help="Disable YOLO object detection.",
    )
    parser.add_argument(
        "--disable-pose",
        action="store_true",
        help="Disable MediaPipe pose movement analysis.",
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

    pipeline = PretrainedClinicalObservationPipeline(
        baseline_model_path=MODEL_PATH,
        action_model_path=ACTION_MODEL_PATH,
        enable_objects=not args.disable_objects,
        enable_pose=not args.disable_pose,
    )
    result = pipeline.predict(video_path)

    print(f"Video: {video_path}")
    print(f"Prediction: {result.predicted_behaviour}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Dangerous objects: {result.dangerous_objects_detected}")
    print(f"Risk group: {result.risk_group}")
    print(f"Risk level: {result.risk_level}")
    print(f"Risk reasons: {result.risk_reasons}")
    print(f"Observation summary: {result.observation_summary}")
    print(f"Alarm required: {result.alarm_required}")
    print(f"Alarm reason: {result.alarm_reason}")
    print(f"Model version: {result.model_version}")
    print(f"Status: {result.status}")
    print(f"Message: {result.message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
