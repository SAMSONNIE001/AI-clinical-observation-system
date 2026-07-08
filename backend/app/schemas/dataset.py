from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from app.domain.enums import BehaviourType, DatasetCategory


class VideoRecordCreate(BaseModel):
    video_id: str
    filename: str
    behaviour_type: BehaviourType
    category: DatasetCategory
    scenario_name: str
    duration_seconds: float = Field(gt=0)
    environment: str
    camera_angle: str | None = None
    recorded_by: str | None = None
    recorded_at: datetime | None = None
    notes: str | None = None


class VideoAnnotationCreate(BaseModel):
    video_id: str
    behaviour: BehaviourType
    start_time_seconds: float = Field(ge=0)
    end_time_seconds: float = Field(gt=0)
    label_quality: float = Field(default=1.0, ge=0.0, le=1.0)
    annotated_by: str
    notes: str | None = None

    @model_validator(mode="after")
    def validate_time_range(self) -> "VideoAnnotationCreate":
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("end_time_seconds must be greater than start_time_seconds")
        return self
