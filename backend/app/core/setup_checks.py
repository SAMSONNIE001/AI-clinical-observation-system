from pathlib import Path

ARTIFACTS = {
    "baseline_model": Path("ml/models/baseline_video_classifier.json"),
    "yolov8_weights": Path("ml/models/yolov8n.pt"),
    "mediapipe_pose_model": Path("ml/models/pose_landmarker_full.task"),
    "action_classifier_checkpoint": Path("ml/models/video_action_grouped_classifier.pt"),
}

OPTIONAL_DEPENDENCIES = [
    ("torch", "PyTorch"),
    ("ultralytics", "Ultralytics YOLO"),
    ("mediapipe", "MediaPipe"),
    ("cv2", "OpenCV"),
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def gather_startup_warnings() -> list[str]:
    warnings: list[str] = []
    warnings.extend(_check_artifacts())
    warnings.extend(_check_optional_dependencies())
    return warnings


def _check_artifacts() -> list[str]:
    warnings: list[str] = []
    for name, path in ARTIFACTS.items():
        resolved = _resolve_path(path)
        if not resolved.exists():
            if name == "baseline_model":
                warnings.append(
                    f"Missing required model artifact: {resolved}. "
                    "Run `python scripts/generate_dataset_metadata.py` and train the baseline model, or provide a baseline model file."
                )
            else:
                warnings.append(
                    f"Missing optional model artifact: {resolved}. "
                    "This feature will remain disabled until the file is available."
                )
    return warnings


def _check_optional_dependencies() -> list[str]:
    warnings: list[str] = []
    for module_name, package_name in OPTIONAL_DEPENDENCIES:
        try:
            __import__(module_name)
        except Exception as exc:
            warnings.append(
                f"Optional dependency not importable: {package_name} ({module_name}). {exc}"
            )
    return warnings
