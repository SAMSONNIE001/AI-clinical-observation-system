import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = PROJECT_ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from fastapi.testclient import TestClient

from app.main import app


def main() -> int:
    client = TestClient(app)
    response = client.post(
        "/api/v1/observation-notes/generate-from-detection",
        json={
            "patient_id": 1,
            "session_id": 100,
            "video_id": "pacing_001",
            "video_path": "dataset/raw/pacing/pacing_001.mp4",
            "camera_id": "test-camera-1",
            "additional_context": "End-to-end detection-to-note workflow test.",
        },
    )
    print(response.status_code)
    payload = response.json()
    print("risk_group:", payload["detection"]["risk_group"])
    print("risk_level:", payload["detection"]["risk_level"])
    print("alarm_required:", payload["detection"]["alarm_required"])
    print("note:", payload["observation_note"]["note"])
    return 0 if response.status_code == 201 else 1


if __name__ == "__main__":
    raise SystemExit(main())
