from fastapi import APIRouter, status

from app.schemas.observation_note import (
    ObservationNoteGenerateRequest,
    ObservationNoteResponse,
    RiskObservationNoteGenerateRequest,
)
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
