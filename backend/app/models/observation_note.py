from datetime import datetime

from pydantic import BaseModel

from app.domain.enums import BehaviourType, RiskLevel


class ObservationNote(BaseModel):
    id: str
    patient_id: int
    session_id: int | None = None
    behaviour: BehaviourType | None = None
    risk_level: RiskLevel
    generated_at: datetime
    note: str
    requires_staff_review: bool = True
    reviewed: bool = False
