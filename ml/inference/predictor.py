from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PredictionResult:
    predicted_behaviour: str | None
    confidence: float
    dangerous_objects_detected: list[str]
    alarm_required: bool
    alarm_reason: str | None
    model_version: str
    status: str
    message: str


class StubBehaviourPredictor:
    model_version = "stub-v0"

    def predict(self, video_path: Path | None = None) -> PredictionResult:
        return PredictionResult(
            predicted_behaviour=None,
            confidence=0.0,
            dangerous_objects_detected=[],
            alarm_required=False,
            alarm_reason=None,
            model_version=self.model_version,
            status="model_not_connected",
            message="Real behaviour detection model is not connected yet.",
        )
