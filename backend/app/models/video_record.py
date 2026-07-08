from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.enums import BehaviourType, DatasetCategory


class VideoRecord(BaseModel):
    id: str
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
    created_at: datetime
