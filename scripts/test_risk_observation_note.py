import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = PROJECT_ROOT / "backend"
for path in (PROJECT_ROOT, BACKEND_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.domain.enums import RiskLevel
from app.schemas.observation_note import RiskObservationNoteGenerateRequest
from app.services.observation_note_service import generate_risk_observation_note


def main() -> int:
    payload = RiskObservationNoteGenerateRequest(
        patient_id=1,
        session_id=100,
        behaviour=None,
        confidence=0.82,
        camera_id="test-camera-1",
        observed_at=datetime.now(timezone.utc),
        alert_generated=True,
        risk_group="object_self_harm_risk",
        risk_level=RiskLevel.HIGH,
        risk_reasons=["Dangerous object cue detected: scissors."],
        observation_summary="Dangerous object cue detected: scissors.",
        dangerous_objects_detected=["scissors"],
        additional_context="Generated from structured risk-engine test.",
    )
    note = generate_risk_observation_note(payload)
    print(note.model_dump())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
