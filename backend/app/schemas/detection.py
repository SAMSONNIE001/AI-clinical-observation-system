from pathlib import Path

from pydantic import BaseModel

from app.domain.enums import BehaviourType


class DetectionRequest(BaseModel):
    video_id: str
    video_path: Path | None = None


class DetectionResponse(BaseModel):
    video_id: str
    predicted_behaviour: BehaviourType | None
    confidence: float
    model_version: str
    status: str
    message: str
