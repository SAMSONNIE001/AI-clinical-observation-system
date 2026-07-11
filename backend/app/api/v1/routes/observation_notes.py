from datetime import datetime, timezone

from fastapi import APIRouter, status

from app.domain.enums import RiskLevel
from app.schemas.detection import DetectionRequest
from app.schemas.observation_note import (
    DetectionObservationNoteGenerateRequest,
    DetectionObservationNoteResponse,
    ObservationNoteGenerateRequest,
    ObservationNoteResponse,
    RiskObservationNoteGenerateRequest,
)
from app.services.detection_service import predict_behaviour
from app.services.observation_note_service import (
    generate_observation_note,
    generate_risk_observation_note,
)


router = APIRouter(prefix="/observation-notes")


@router.post(
    "/generate",
    response_model=ObservationNoteResponse,
    status_code=status.HTTP_201_CREATED,
)
def generate_note(
    payload: ObservationNoteGenerateRequest,
) -> ObservationNoteResponse:
    return generate_observation_note(payload)


@router.post(
    "/generate-from-risk",
    response_model=ObservationNoteResponse,
    status_code=status.HTTP_201_CREATED,
)
def generate_note_from_risk(
    payload: RiskObservationNoteGenerateRequest,
) -> ObservationNoteResponse:
    return generate_risk_observation_note(payload)


@router.post(
    "/generate-from-detection",
    response_model=DetectionObservationNoteResponse,
    status_code=status.HTTP_201_CREATED,
)
def generate_note_from_detection(
    payload: DetectionObservationNoteGenerateRequest,
) -> DetectionObservationNoteResponse:
    detection = predict_behaviour(
        DetectionRequest(
            video_id=payload.video_id,
            video_path=payload.video_path,
        )
    )
    risk_payload = RiskObservationNoteGenerateRequest(
        patient_id=payload.patient_id,
        session_id=payload.session_id,
        behaviour=detection.predicted_behaviour,
        confidence=detection.confidence,
        camera_id=payload.camera_id,
        observed_at=payload.observed_at or datetime.now(timezone.utc),
        alert_generated=detection.alarm_required,
        risk_group=detection.risk_group,
        risk_level=_risk_level_from_detection(detection.risk_level),
        risk_reasons=detection.risk_reasons,
        observation_summary=detection.observation_summary,
        dangerous_objects_detected=detection.dangerous_objects_detected,
        additional_context=payload.additional_context,
    )
    note = generate_risk_observation_note(risk_payload)
    return DetectionObservationNoteResponse(
        detection=detection,
        observation_note=ObservationNoteResponse(**note.model_dump()),
    )


def _risk_level_from_detection(risk_level: str) -> RiskLevel:
    try:
        return RiskLevel(risk_level)
    except ValueError:
        return RiskLevel.LOW
