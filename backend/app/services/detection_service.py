from app.domain.risk import alarm_required_for_behaviour, alarm_required_for_objects
from app.schemas.detection import DetectionRequest, DetectionResponse


STUB_MODEL_VERSION = "stub-v0"


def predict_behaviour(payload: DetectionRequest) -> DetectionResponse:
    predicted_behaviour = None
    dangerous_objects_detected: list[str] = []
    alarm_required = (
        alarm_required_for_behaviour(predicted_behaviour)
        or alarm_required_for_objects(dangerous_objects_detected)
    )

    return DetectionResponse(
        video_id=payload.video_id,
        predicted_behaviour=predicted_behaviour,
        confidence=0.0,
        dangerous_objects_detected=dangerous_objects_detected,
        alarm_required=alarm_required,
        alarm_reason=(
            "High-risk behaviour or dangerous object detected."
            if alarm_required
            else None
        ),
        model_version=STUB_MODEL_VERSION,
        status="model_not_connected",
        message="Detection pipeline contract is ready; real ML model is not connected yet.",
    )
