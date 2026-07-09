from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.enums import BehaviourType
from app.domain.risk import alarm_required_for_behaviour


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

    def model_post_init(self, __context: object) -> None:
        if alarm_required_for_behaviour(self.behaviour):
            self.alert_generated = True
