from app.domain.enums import BehaviourType, RiskLevel


HIGH_RISK_BEHAVIOURS = {
    BehaviourType.FALL,
    BehaviourType.FIGHTING,
    BehaviourType.ATTACK_ON_PERSON,
    BehaviourType.HEAD_BANGING,
    BehaviourType.PROPERTY_DAMAGE,
    BehaviourType.CHOKING_SIMULATION,
    BehaviourType.LIGATURE_RISK,
    BehaviourType.CUTTING_RISK,
    BehaviourType.BLEEDING_VISIBLE,
    BehaviourType.BLOOD_VISIBLE,
    BehaviourType.SHARP_OBJECT_DETECTED,
}

HIGH_RISK_OBJECTS = {
    "blade",
    "box_cutter",
    "broken_glass",
    "knife",
    "razor",
    "scissors",
    "sharp_object",
}

MEDIUM_RISK_BEHAVIOURS = {
    BehaviourType.PACING,
    BehaviourType.AGGRESSIVE_MOVEMENT,
    BehaviourType.PROLONGED_INACTIVITY,
}


def risk_level_for_behaviour(behaviour: BehaviourType | None) -> RiskLevel:
    if behaviour in HIGH_RISK_BEHAVIOURS:
        return RiskLevel.HIGH
    if behaviour in MEDIUM_RISK_BEHAVIOURS:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def alarm_required_for_behaviour(behaviour: BehaviourType | None) -> bool:
    return risk_level_for_behaviour(behaviour) == RiskLevel.HIGH


def alarm_required_for_objects(detected_objects: list[str]) -> bool:
    normalised_objects = {item.strip().lower() for item in detected_objects}
    return bool(normalised_objects & HIGH_RISK_OBJECTS)
