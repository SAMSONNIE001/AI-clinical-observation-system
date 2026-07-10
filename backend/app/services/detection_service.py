import sys
from functools import lru_cache
from pathlib import Path

from app.domain.enums import BehaviourType
from app.domain.risk import alarm_required_for_behaviour, alarm_required_for_objects
from app.schemas.detection import DetectionRequest, DetectionResponse


STUB_MODEL_VERSION = "stub-v0"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "baseline_video_classifier.json"


def predict_behaviour(payload: DetectionRequest) -> DetectionResponse:
    if payload.video_path is None:
        return _stub_response(
            payload=payload,
            status="missing_video_path",
            message="A video_path is required for baseline video classification.",
        )

    video_path = _resolve_video_path(payload.video_path)
    if not video_path.exists():
        return _stub_response(
            payload=payload,
            status="video_not_found",
            message=f"Video file not found: {payload.video_path}",
        )

    if not BASELINE_MODEL_PATH.exists():
        return _stub_response(
            payload=payload,
            status="model_not_found",
            message=f"Baseline model file not found: {BASELINE_MODEL_PATH}",
        )

    try:
        prediction = _baseline_predictor().predict(video_path)
    except Exception as exc:
        return _stub_response(
            payload=payload,
            status="prediction_failed",
            message=f"Baseline prediction failed: {exc}",
        )

    predicted_behaviour = (
        BehaviourType(prediction.predicted_behaviour)
        if prediction.predicted_behaviour is not None
        else None
    )
    dangerous_objects_detected, object_detector_status = _detect_dangerous_objects(
        video_path
    )
    object_alarm_required = alarm_required_for_objects(dangerous_objects_detected)
    alarm_required = prediction.alarm_required or object_alarm_required
    alarm_reason = prediction.alarm_reason
    if object_alarm_required:
        alarm_reason = "Dangerous object detected."
    if prediction.alarm_required and object_alarm_required:
        alarm_reason = "High-risk behaviour and dangerous object detected."

    model_version = prediction.model_version
    if object_detector_status == "ok":
        model_version = f"{model_version}+yolo-object-v1"

    message = prediction.message
    if object_detector_status != "ok":
        message = f"{message} Object detector status: {object_detector_status}."

    return DetectionResponse(
        video_id=payload.video_id,
        predicted_behaviour=predicted_behaviour,
        confidence=prediction.confidence,
        dangerous_objects_detected=dangerous_objects_detected,
        alarm_required=alarm_required,
        alarm_reason=alarm_reason,
        model_version=model_version,
        status=prediction.status,
        message=message,
    )


def _stub_response(
    payload: DetectionRequest,
    status: str = "model_not_connected",
    message: str = "Detection pipeline contract is ready; real ML model is not connected yet.",
) -> DetectionResponse:
    dangerous_objects_detected: list[str] = []
    alarm_required = (
        alarm_required_for_behaviour(None)
        or alarm_required_for_objects(dangerous_objects_detected)
    )

    return DetectionResponse(
        video_id=payload.video_id,
        predicted_behaviour=None,
        confidence=0.0,
        dangerous_objects_detected=dangerous_objects_detected,
        alarm_required=alarm_required,
        alarm_reason=(
            "High-risk behaviour or dangerous object detected."
            if alarm_required
            else None
        ),
        model_version=STUB_MODEL_VERSION,
        status=status,
        message=message,
    )


def _resolve_video_path(video_path: Path) -> Path:
    if video_path.is_absolute():
        return video_path
    return PROJECT_ROOT / video_path


@lru_cache(maxsize=1)
def _baseline_predictor():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from ml.inference.predictor import BaselineVideoClassifierPredictor

    return BaselineVideoClassifierPredictor(model_path=BASELINE_MODEL_PATH)


def _detect_dangerous_objects(video_path: Path) -> tuple[list[str], str]:
    try:
        from ml.inference.object_detector import dangerous_objects_from_detections

        detections = _pretrained_object_detector().detect_video(video_path)
        return dangerous_objects_from_detections(detections), "ok"
    except Exception as exc:
        return [], f"unavailable ({exc})"


@lru_cache(maxsize=1)
def _pretrained_object_detector():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from ml.inference.object_detector import PretrainedObjectDetector

    return PretrainedObjectDetector()
