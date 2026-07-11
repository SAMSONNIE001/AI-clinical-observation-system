from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from app.domain.enums import BehaviourType, RiskLevel
from app.schemas.detection import DetectionResponse


class ObservationNoteGenerateRequest(BaseModel):
    patient_id: int
    session_id: int | None = None
    behaviour: BehaviourType
    confidence: float = Field(ge=0.0, le=1.0)
    camera_id: str | None = None
    observed_at: datetime
    alert_generated: bool = False
    additional_context: str | None = None


class RiskObservationNoteGenerateRequest(BaseModel):
    patient_id: int
    session_id: int | None = None
    behaviour: BehaviourType | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    camera_id: str | None = None
    observed_at: datetime
    alert_generated: bool = False
    risk_group: str | None = None
    risk_level: RiskLevel = RiskLevel.LOW
    risk_reasons: list[str] = Field(default_factory=list)
    observation_summary: str | None = None
    dangerous_objects_detected: list[str] = Field(default_factory=list)
    additional_context: str | None = None


class DetectionObservationNoteGenerateRequest(BaseModel):
    patient_id: int
    video_id: str
    video_path: Path | None = None
    session_id: int | None = None
    camera_id: str | None = None
    observed_at: datetime | None = None
    additional_context: str | None = None


class ObservationNoteResponse(BaseModel):
    id: str
    patient_id: int
    session_id: int | None
    behaviour: BehaviourType | None
    risk_group: str | None = None
    risk_level: RiskLevel
    risk_reasons: list[str] = Field(default_factory=list)
    observation_summary: str | None = None
    generated_at: datetime
    note: str
    requires_staff_review: bool
    reviewed: bool


class DetectionObservationNoteResponse(BaseModel):
    detection: DetectionResponse
    observation_note: ObservationNoteResponse
