from dataclasses import dataclass, field

from ml.inference.pose_detector import PoseMovementSummary
from ml.training.label_groups import (
    MEDICAL_SAFETY_RISK,
    MOVEMENT_AGITATION,
    NORMAL_ACTIVITY,
    OBJECT_SELF_HARM_RISK,
    VIOLENCE_PROPERTY_RISK,
)


HIGH_RISK_GROUPS = {
    MEDICAL_SAFETY_RISK,
    OBJECT_SELF_HARM_RISK,
    VIOLENCE_PROPERTY_RISK,
}


@dataclass(frozen=True)
class RiskSignalSnapshot:
    action_group: str | None = None
    action_confidence: float = 0.0
    dangerous_objects: list[str] = field(default_factory=list)
    clinical_object_cues: list[str] = field(default_factory=list)
    pose_summary: PoseMovementSummary | None = None


@dataclass(frozen=True)
class RiskAssessment:
    risk_group: str
    risk_level: str
    alarm_required: bool
    reasons: list[str]
    observation_summary: str


def assess_risk(signals: RiskSignalSnapshot) -> RiskAssessment:
    reasons: list[str] = []
    candidate_groups: list[tuple[str, str, str]] = []
    pose_group = _pose_risk_group(signals.pose_summary)

    if signals.dangerous_objects and pose_group is not None:
        objects = ", ".join(signals.dangerous_objects)
        candidate_groups.append(
            (
                OBJECT_SELF_HARM_RISK,
                "high",
                f"Dangerous object cue detected with body movement signal: {objects}.",
            )
        )

    if signals.dangerous_objects:
        objects = ", ".join(signals.dangerous_objects)
        candidate_groups.append(
            (
                OBJECT_SELF_HARM_RISK,
                "high",
                f"Dangerous object cue detected: {objects}.",
            )
        )

    if signals.clinical_object_cues and pose_group is not None:
        cues = ", ".join(signals.clinical_object_cues)
        candidate_groups.append(
            (
                OBJECT_SELF_HARM_RISK,
                "medium",
                f"Clinical object caution cue detected with body movement signal: {cues}.",
            )
        )

    if signals.clinical_object_cues:
        cues = ", ".join(signals.clinical_object_cues)
        candidate_groups.append(
            (
                OBJECT_SELF_HARM_RISK,
                "medium",
                f"Clinical object caution cue detected: {cues}.",
            )
        )

    if signals.action_group in HIGH_RISK_GROUPS:
        candidate_groups.append(
            (
                signals.action_group,
                "high",
                f"Action model predicted high-risk group: {signals.action_group}.",
            )
        )
    elif signals.action_group == MOVEMENT_AGITATION:
        candidate_groups.append(
            (
                MOVEMENT_AGITATION,
                "medium",
                "Action model predicted movement or agitation group.",
            )
        )

    if pose_group is not None:
        candidate_groups.append(pose_group)

    if not candidate_groups:
        return RiskAssessment(
            risk_group=signals.action_group or NORMAL_ACTIVITY,
            risk_level="low",
            alarm_required=False,
            reasons=[],
            observation_summary="No high-risk visual cue detected.",
        )

    risk_group, risk_level, primary_reason = max(
        candidate_groups,
        key=lambda item: _risk_rank(item[1]),
    )
    reasons = [reason for _, _, reason in candidate_groups]
    alarm_required = risk_level == "high"

    return RiskAssessment(
        risk_group=risk_group,
        risk_level=risk_level,
        alarm_required=alarm_required,
        reasons=reasons,
        observation_summary=primary_reason,
    )


def _pose_risk_group(
    pose_summary: PoseMovementSummary | None,
) -> tuple[str, str, str] | None:
    if pose_summary is None or pose_summary.pose_coverage < 0.4:
        return None

    if pose_summary.posture_change >= 0.22:
        return (
            MEDICAL_SAFETY_RISK,
            "medium",
            "Large posture change detected.",
        )

    if pose_summary.mean_motion >= 0.055 or pose_summary.max_motion >= 0.35:
        return (
            MOVEMENT_AGITATION,
            "medium",
            "High body movement detected from pose landmarks.",
        )

    if pose_summary.mean_motion <= 0.006 and pose_summary.pose_coverage >= 0.7:
        return (
            MEDICAL_SAFETY_RISK,
            "medium",
            "Very low movement detected across sampled frames.",
        )

    return None


def _risk_rank(risk_level: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(risk_level, 0)
