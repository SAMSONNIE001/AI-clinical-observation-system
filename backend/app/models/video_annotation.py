from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from app.domain.enums import BehaviourType


class VideoAnnotation(BaseModel):
    id: str
    video_id: str
    behaviour: BehaviourType
    start_time_seconds: float = Field(ge=0)
    end_time_seconds: float = Field(gt=0)
    label_quality: float = Field(default=1.0, ge=0.0, le=1.0)
    annotated_by: str
    notes: str | None = None
    created_at: datetime

    @model_validator(mode="after")
    def validate_time_range(self) -> "VideoAnnotation":
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("end_time_seconds must be greater than start_time_seconds")
        return self
