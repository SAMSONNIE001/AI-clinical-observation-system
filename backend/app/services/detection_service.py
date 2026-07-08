from app.schemas.detection import DetectionRequest, DetectionResponse


STUB_MODEL_VERSION = "stub-v0"


def predict_behaviour(payload: DetectionRequest) -> DetectionResponse:
    return DetectionResponse(
        video_id=payload.video_id,
        predicted_behaviour=None,
        confidence=0.0,
        model_version=STUB_MODEL_VERSION,
        status="model_not_connected",
        message="Detection pipeline contract is ready; real ML model is not connected yet.",
    )
