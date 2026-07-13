import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = PROJECT_ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from app.main import app
from app.schemas.dataset import VideoRecordCreate
from app.services.dataset_service import create_video_record, export_training_manifest


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_root_route(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "AI Clinical Observation System API is running"})

    def test_health_route(self) -> None:
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_generate_observation_note(self) -> None:
        payload = {
            "patient_id": 1,
            "session_id": 10,
            "behaviour": "walking",
            "confidence": 0.78,
            "observed_at": "2026-01-01T12:00:00Z",
            "alert_generated": False,
        }
        response = self.client.post("/api/v1/observation-notes/generate", json=payload)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["patient_id"], 1)
        self.assertEqual(data["behaviour"], "walking")
        self.assertIn("note", data)

    def test_detection_predict_missing_video(self) -> None:
        response = self.client.post(
            "/api/v1/detection/predict",
            json={"video_id": "missing_video", "video_path": "nonexistent.mp4"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "video_not_found")

    def test_generate_from_detection(self) -> None:
        payload = {
            "patient_id": 1,
            "video_id": "pacing_001",
            "video_path": "dataset/raw/pacing/pacing_001.mp4",
            "camera_id": "camera-1",
            "additional_context": "Test note generation from detection.",
        }
        response = self.client.post("/api/v1/observation-notes/generate-from-detection", json=payload)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("detection", data)
        self.assertIn("observation_note", data)
        self.assertEqual(data["observation_note"]["patient_id"], 1)


class DatasetServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = TemporaryDirectory()
        metadata_dir = Path(self.tempdir.name) / "dataset"
        export_dir = Path(self.tempdir.name) / "exports"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)
        self.fake_settings = SimpleNamespace(
            dataset_metadata_dir=metadata_dir,
            dataset_export_dir=export_dir,
        )
        self.settings_patcher = patch("app.services.dataset_service.settings", new=self.fake_settings)
        self.settings_patcher.start()

    def tearDown(self) -> None:
        self.settings_patcher.stop()
        self.tempdir.cleanup()

    def test_create_and_export_video_record(self) -> None:
        payload = VideoRecordCreate(
            video_id="test_video_001",
            filename="test_video.mp4",
            behaviour_type="walking",
            category="normal",
            scenario_name="test scenario",
            duration_seconds=4.5,
            environment="indoor",
            camera_angle="front",
        )
        record = create_video_record(payload)
        self.assertEqual(record.video_id, "test_video_001")

        export = export_training_manifest()
        self.assertEqual(export.record_count, 1)
        self.assertTrue(export.export_path.exists())

    def test_export_no_records_raises(self) -> None:
        from app.services.dataset_service import export_training_manifest

        with self.assertRaises(Exception):
            export_training_manifest()
