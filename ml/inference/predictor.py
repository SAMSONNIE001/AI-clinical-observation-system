from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PredictionResult:
    predicted_behaviour: str | None
    confidence: float
    model_version: str
    status: str
    message: str


class StubBehaviourPredictor:
    model_version = "stub-v0"

    def predict(self, video_path: Path | None = None) -> PredictionResult:
        return PredictionResult(
            predicted_behaviour=None,
            confidence=0.0,
            model_version=self.model_version,
            status="model_not_connected",
            message="Real behaviour detection model is not connected yet.",
        )
