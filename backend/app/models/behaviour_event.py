from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.enums import BehaviourType


class BehaviourEvent(BaseModel):
    id: int
    patient_id: int
    session_id: int
    behaviour: BehaviourType
    confidence: float = Field(ge=0.0, le=1.0)
    camera_id: str
    timestamp: datetime
    reviewed: bool = False
    alert_generated: bool = False
