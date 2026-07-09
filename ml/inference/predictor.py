import json
from dataclasses import dataclass
from pathlib import Path

from app.domain.enums import BehaviourType
from app.domain.risk import alarm_required_for_behaviour
from ml.training.train_baseline_classifier import predict_with_model
from ml.training.video_features import extract_video_features


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


class BaselineVideoClassifierPredictor:
    def __init__(
        self,
        model_path: Path = Path("ml/models/baseline_video_classifier.json"),
    ) -> None:
        self.model_path = model_path
        with model_path.open("r", encoding="utf-8") as model_file:
            self.model = json.load(model_file)
        self.model_version = f"baseline-{self.model['feature_version']}"

    def predict(self, video_path: Path | None = None) -> PredictionResult:
        if video_path is None:
            return PredictionResult(
                predicted_behaviour=None,
                confidence=0.0,
                dangerous_objects_detected=[],
                alarm_required=False,
                alarm_reason=None,
                model_version=self.model_version,
                status="missing_video_path",
                message="A video_path is required for baseline video classification.",
            )

        features = extract_video_features(video_path)
        label, confidence = predict_with_model(self.model, features)
        behaviour = BehaviourType(label)
        alarm_required = alarm_required_for_behaviour(behaviour)

        return PredictionResult(
            predicted_behaviour=label,
            confidence=confidence,
            dangerous_objects_detected=[],
            alarm_required=alarm_required,
            alarm_reason="High-risk behaviour detected." if alarm_required else None,
            model_version=self.model_version,
            status="ok",
            message="Baseline video classifier prediction completed.",
        )
