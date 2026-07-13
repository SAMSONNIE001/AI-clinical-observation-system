import io
import os
import sys
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
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

    @patch("app.services.observation_note_service.openai.ChatCompletion.create")
    def test_generate_observation_note_uses_llm(self, mock_chat_completion) -> None:
        mock_chat_completion.return_value = {
            "choices": [
                {"message": {"content": "LLM draft note: Patient 1 was observed walking."}}
            ]
        }
        payload = {
            "patient_id": 1,
            "session_id": 10,
            "behaviour": "walking",
            "confidence": 0.85,
            "observed_at": "2026-01-01T12:30:00Z",
            "alert_generated": False,
        }

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}, clear=False):
            with patch("app.services.observation_note_service.USE_LLM_NOTE_GEN", True):
                response = self.client.post("/api/v1/observation-notes/generate", json=payload)

        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["patient_id"], 1)
        self.assertEqual(data["behaviour"], "walking")
        self.assertEqual(data["note"], "LLM draft note: Patient 1 was observed walking.")
        self.assertTrue(mock_chat_completion.called)

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

    def test_video_upload(self) -> None:
        with NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(b"fakevideo")
            temp_file.flush()
            with open(temp_file.name, "rb") as payload_file:
                response = self.client.post(
                    "/api/v1/videos/upload",
                    files={"file": (Path(temp_file.name).name, payload_file, "video/mp4")},
                )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["original_filename"], Path(temp_file.name).name)
        self.assertTrue(data["stored_filename"].endswith(".mp4"))


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

    def test_video_record_and_annotation_routes(self) -> None:
        from app.schemas.dataset import VideoAnnotationCreate

        record_payload = VideoRecordCreate(
            video_id="video_002",
            filename="video_002.mp4",
            behaviour_type="sitting",
            category="normal",
            scenario_name="test scenario",
            duration_seconds=5.0,
            environment="ward",
        )
        create_video_record(record_payload)

        annotation_payload = VideoAnnotationCreate(
            video_id="video_002",
            behaviour="sitting",
            start_time_seconds=0.5,
            end_time_seconds=3.0,
            annotated_by="tester",
        )
        from app.services.dataset_service import create_video_annotation, list_video_annotations

        annotation = create_video_annotation(annotation_payload)
        self.assertEqual(annotation.video_id, "video_002")
        annotations = list_video_annotations(video_id="video_002")
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].behaviour, "sitting")
