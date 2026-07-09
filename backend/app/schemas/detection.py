from pathlib import Path

from pydantic import BaseModel, Field

from app.domain.enums import BehaviourType


class DetectionRequest(BaseModel):
    video_id: str
    video_path: Path | None = None


class DetectionResponse(BaseModel):
    video_id: str
    predicted_behaviour: BehaviourType | None
    confidence: float
    dangerous_objects_detected: list[str] = Field(default_factory=list)
    alarm_required: bool = False
    alarm_reason: str | None = None
    model_version: str
    status: str
    message: str
