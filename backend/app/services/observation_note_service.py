import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore[assignment]
    OPENAI_AVAILABLE = False

from app.domain.enums import BehaviourType, RiskLevel
from app.domain.risk import risk_level_for_behaviour
from app.models.observation_note import ObservationNote
from app.schemas.observation_note import (
    ObservationNoteGenerateRequest,
    RiskObservationNoteGenerateRequest,
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _load_openai_api_key() -> str | None:
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key

    key_path = BACKEND_ROOT / "apikey"
    if not key_path.exists():
        return None

    key_files: list[Path] = []
    if key_path.is_file():
        key_files = [key_path]
    elif key_path.is_dir():
        key_files = sorted(
            [path for path in key_path.iterdir() if path.is_file()],
            key=lambda p: p.name,
        )
    if not key_files:
        return None

    for candidate in key_files:
        try:
            key = candidate.read_text(encoding="utf-8").strip()
            if key:
                os.environ["OPENAI_API_KEY"] = key
                return key
        except OSError:
            continue

    return None

OPENAI_API_KEY = _load_openai_api_key()
USE_LLM_NOTE_GEN = bool(OPENAI_API_KEY and OPENAI_AVAILABLE)
if USE_LLM_NOTE_GEN and openai is not None:
    openai.api_key = OPENAI_API_KEY


def generate_observation_note(payload: ObservationNoteGenerateRequest) -> ObservationNote:
    risk_level = _risk_level_for(payload.behaviour, payload.alert_generated)
    note = _build_note_text(payload, risk_level)

    llm_note = _build_llm_note_text(payload, note)
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
        note=llm_note or note,
        requires_staff_review=True,
        reviewed=False,
    )


def generate_risk_observation_note(
    payload: RiskObservationNoteGenerateRequest,
) -> ObservationNote:
    note = _build_risk_note_text(payload)
    llm_note = _build_llm_note_text(payload, note)
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
        note=llm_note or note,
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


def _build_llm_note_text(
    payload: ObservationNoteGenerateRequest | RiskObservationNoteGenerateRequest,
    fallback_note: str,
) -> str | None:
    if not USE_LLM_NOTE_GEN:
        return None

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical observation assistant that writes concise draft observation notes "
                    "for healthcare staff. The output should be a single, readable note suitable for review."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create a draft observation note from the following structured information:\n"
                    f"Patient ID: {payload.patient_id}\n"
                    f"Session ID: {payload.session_id}\n"
                    f"Behaviour: {payload.behaviour or 'unknown'}\n"
                    f"Confidence: {round(payload.confidence * 100)}%\n"
                    f"Observed at: {payload.observed_at.isoformat()}\n"
                    f"Camera ID: {payload.camera_id or 'none'}\n"
                    f"Alert generated: {payload.alert_generated}\n"
                    f"Additional context: {payload.additional_context or 'none'}\n"
                ),
            },
        ]

        if isinstance(payload, RiskObservationNoteGenerateRequest):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Risk group: {payload.risk_group or 'none'}\n"
                        f"Risk level: {payload.risk_level.value}\n"
                        f"Risk reasons: {', '.join(payload.risk_reasons) or 'none'}\n"
                        f"Observation summary: {payload.observation_summary or 'none'}\n"
                        f"Dangerous objects: {', '.join(payload.dangerous_objects_detected) or 'none'}"
                    ),
                }
            )

        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=300,
        )
        choices = response.get("choices")
        if not choices:
            return None
        note_text = choices[0].get("message", {}).get("content")
        return note_text.strip() if note_text else None
    except Exception:
        return None


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
