import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_PATH = PROJECT_ROOT / "ml" / "models" / "yolov8n.pt"
MEDIAPIPE_PATH = PROJECT_ROOT / "ml" / "models" / "pose_landmarker_full.task"

YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def download(url: str, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Already present: {output_path}")
        return 0

    print(f"Downloading {url} to {output_path}")
    try:
        urllib.request.urlretrieve(url, output_path)
        print("Download complete.")
        return 0
    except Exception as exc:
        print(f"Failed to download {url}: {exc}")
        return 1


def main() -> int:
    print("Downloading optional model artifacts...")
    exit_code = download(YOLO_URL, YOLO_PATH)
    if exit_code != 0:
        return exit_code
    return download(POSE_LANDMARKER_URL, MEDIAPIPE_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
