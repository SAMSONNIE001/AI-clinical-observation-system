from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.enums import BehaviourType, RiskLevel


class ObservationNoteGenerateRequest(BaseModel):
    patient_id: int
    session_id: int | None = None
    behaviour: BehaviourType
    confidence: float = Field(ge=0.0, le=1.0)
    camera_id: str | None = None
    observed_at: datetime
    alert_generated: bool = False
    additional_context: str | None = None


class ObservationNoteResponse(BaseModel):
    id: str
    patient_id: int
    session_id: int | None
    behaviour: BehaviourType | None
    risk_level: RiskLevel
    generated_at: datetime
    note: str
    requires_staff_review: bool
    reviewed: bool
