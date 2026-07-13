import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = PROJECT_ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from app.core.setup_checks import gather_startup_warnings
from app.services.observation_note_service import BACKEND_ROOT, _load_openai_api_key


class ModelSetupTests(unittest.TestCase):
    def test_setup_warnings_include_optional_dependencies(self) -> None:
        warnings = gather_startup_warnings()
        self.assertIsInstance(warnings, list)
        self.assertTrue(all(isinstance(w, str) for w in warnings))

    @patch("builtins.__import__")
    def test_missing_optional_dependency_warning(self, mock_import):
        original_import = __import__

        def side_effect(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "mediapipe":
                raise ImportError("No module named mediapipe")
            return original_import(name, globals, locals, fromlist, level)

        mock_import.side_effect = side_effect
        warnings = gather_startup_warnings()
        self.assertTrue(any("MediaPipe" in warning for warning in warnings))

    def test_load_openai_api_key_from_file(self) -> None:
        import tempfile

        original_env = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                temp_path = Path(tempdir) / "apikey"
                temp_path.write_text("test-openai-key\n", encoding="utf-8")
                with patch("app.services.observation_note_service.BACKEND_ROOT", Path(tempdir)):
                    key = _load_openai_api_key()
                self.assertEqual(key, "test-openai-key")
                self.assertEqual(os.environ.get("OPENAI_API_KEY"), "test-openai-key")
        finally:
            if original_env is not None:
                os.environ["OPENAI_API_KEY"] = original_env
