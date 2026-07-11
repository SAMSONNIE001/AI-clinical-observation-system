import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "ml" / "models" / "yolov8n.pt"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        print(f"YOLO weights already exist: {OUTPUT_PATH}")
        return 0

    print(f"Downloading YOLO weights from {YOLO_URL}")
    print(f"Saving to {OUTPUT_PATH}")
    urllib.request.urlretrieve(YOLO_URL, OUTPUT_PATH)
    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
