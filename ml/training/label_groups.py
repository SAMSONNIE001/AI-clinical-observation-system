NORMAL_ACTIVITY = "normal_activity"
MOVEMENT_AGITATION = "movement_agitation"
MEDICAL_SAFETY_RISK = "medical_safety_risk"
VIOLENCE_PROPERTY_RISK = "violence_property_risk"
OBJECT_SELF_HARM_RISK = "object_self_harm_risk"


LABEL_GROUPS = {
    "sleeping": NORMAL_ACTIVITY,
    "sitting": NORMAL_ACTIVITY,
    "standing": NORMAL_ACTIVITY,
    "walking": NORMAL_ACTIVITY,
    "eating": NORMAL_ACTIVITY,
    "reading": NORMAL_ACTIVITY,
    "pacing": MOVEMENT_AGITATION,
    "aggressive_movement": MOVEMENT_AGITATION,
    "prolonged_inactivity": MEDICAL_SAFETY_RISK,
    "fall": MEDICAL_SAFETY_RISK,
    "choking_simulation": MEDICAL_SAFETY_RISK,
    "vomiting": MEDICAL_SAFETY_RISK,
    "attack_on_person": VIOLENCE_PROPERTY_RISK,
    "head_banging": VIOLENCE_PROPERTY_RISK,
    "property_damage": VIOLENCE_PROPERTY_RISK,
    "ligature_risk": OBJECT_SELF_HARM_RISK,
    "bleeding_visible": OBJECT_SELF_HARM_RISK,
    "sharp_object_detected": OBJECT_SELF_HARM_RISK,
}


HIGH_RISK_GROUPS = {
    MEDICAL_SAFETY_RISK,
    VIOLENCE_PROPERTY_RISK,
    OBJECT_SELF_HARM_RISK,
}


def grouped_label(label: str) -> str:
    return LABEL_GROUPS.get(label, label)


def action_group_alarm_required(label: str | None) -> bool:
    return label in HIGH_RISK_GROUPS
