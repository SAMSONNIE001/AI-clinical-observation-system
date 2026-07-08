from fastapi import APIRouter

from app.schemas.detection import DetectionRequest, DetectionResponse
from app.services.detection_service import predict_behaviour


router = APIRouter(prefix="/detection")


@router.post("/predict", response_model=DetectionResponse)
def predict_video_behaviour(payload: DetectionRequest) -> DetectionResponse:
    return predict_behaviour(payload)
