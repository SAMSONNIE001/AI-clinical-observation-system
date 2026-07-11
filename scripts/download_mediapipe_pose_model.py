import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "ml" / "models" / "pose_landmarker_full.task"
POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        print(f"MediaPipe pose model already exists: {OUTPUT_PATH}")
        return 0

    print(f"Downloading MediaPipe pose model from {POSE_LANDMARKER_URL}")
    print(f"Saving to {OUTPUT_PATH}")
    urllib.request.urlretrieve(POSE_LANDMARKER_URL, OUTPUT_PATH)
    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
