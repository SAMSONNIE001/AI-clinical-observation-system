from datetime import datetime, timezone
from uuid import uuid4

from app.domain.enums import BehaviourType, RiskLevel
from app.domain.risk import risk_level_for_behaviour
from app.models.observation_note import ObservationNote
from app.schemas.observation_note import (
    ObservationNoteGenerateRequest,
    RiskObservationNoteGenerateRequest,
)


def generate_observation_note(payload: ObservationNoteGenerateRequest) -> ObservationNote:
    risk_level = _risk_level_for(payload.behaviour, payload.alert_generated)
    note = _build_note_text(payload, risk_level)

    return ObservationNote(
        id=str(uuid4()),
        patient_id=payload.patient_id,
        session_id=payload.session_id,
        behaviour=payload.behaviour,
        risk_group=None,
        risk_level=risk_level,
        risk_reasons=[],
        observation_summary=None,
        generated_at=datetime.now(timezone.utc),
        note=note,
        requires_staff_review=True,
        reviewed=False,
    )


def generate_risk_observation_note(
    payload: RiskObservationNoteGenerateRequest,
) -> ObservationNote:
    note = _build_risk_note_text(payload)

    return ObservationNote(
        id=str(uuid4()),
        patient_id=payload.patient_id,
        session_id=payload.session_id,
        behaviour=payload.behaviour,
        risk_group=payload.risk_group,
        risk_level=payload.risk_level,
        risk_reasons=payload.risk_reasons,
        observation_summary=payload.observation_summary,
        generated_at=datetime.now(timezone.utc),
        note=note,
        requires_staff_review=True,
        reviewed=False,
    )


def _risk_level_for(behaviour: BehaviourType, alert_generated: bool) -> RiskLevel:
    if alert_generated:
        return RiskLevel.HIGH
    return risk_level_for_behaviour(behaviour)


def _build_note_text(
    payload: ObservationNoteGenerateRequest,
    risk_level: RiskLevel,
) -> str:
    behaviour = payload.behaviour.value.replace("_", " ")
    confidence_percent = round(payload.confidence * 100)
    observed_at = payload.observed_at.isoformat()

    camera_text = f" on camera {payload.camera_id}" if payload.camera_id else ""
    alert_text = " An alert was generated." if payload.alert_generated else ""
    context_text = (
        f" Additional context: {payload.additional_context}"
        if payload.additional_context
        else ""
    )

    return (
        f"Draft observation note: Patient {payload.patient_id} was observed"
        f"{camera_text} at {observed_at} with behaviour labelled as {behaviour}."
        f" Confidence: {confidence_percent}%. Risk level: {risk_level.value}."
        f"{alert_text}{context_text} Staff review and confirmation required."
    )


def _build_risk_note_text(payload: RiskObservationNoteGenerateRequest) -> str:
    observed_at = payload.observed_at.isoformat()
    confidence_percent = round(payload.confidence * 100)
    behaviour_text = (
        payload.behaviour.value.replace("_", " ")
        if payload.behaviour
        else "not confirmed"
    )
    risk_group_text = (
        payload.risk_group.replace("_", " ")
        if payload.risk_group
        else "not assigned"
    )
    camera_text = f" on camera {payload.camera_id}" if payload.camera_id else ""
    alert_text = " An alert was generated." if payload.alert_generated else ""
    summary_text = (
        f" Summary: {payload.observation_summary}"
        if payload.observation_summary
        else ""
    )
    reasons_text = (
        " Risk reasons: "
        + "; ".join(_strip_sentence_end(reason) for reason in payload.risk_reasons)
        + "."
        if payload.risk_reasons
        else ""
    )
    objects_text = (
        " Dangerous object cues: "
        + ", ".join(payload.dangerous_objects_detected)
        + "."
        if payload.dangerous_objects_detected
        else ""
    )
    context_text = (
        f" Additional context: {payload.additional_context}"
        if payload.additional_context
        else ""
    )

    return (
        f"Draft observation note: Patient {payload.patient_id} was observed"
        f"{camera_text} at {observed_at}. Behaviour: {behaviour_text}."
        f" Structured risk group: {risk_group_text}. Risk level:"
        f" {payload.risk_level.value}. Confidence: {confidence_percent}%."
        f"{summary_text}{reasons_text}{objects_text}{alert_text}{context_text}"
        " Staff review and confirmation required."
    )


def _strip_sentence_end(text: str) -> str:
    return text.rstrip().rstrip(".")
