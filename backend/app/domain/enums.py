from enum import Enum


class BehaviourType(str, Enum):
    SLEEPING = "sleeping"
    SITTING = "sitting"
    STANDING = "standing"
    WALKING = "walking"
    EATING = "eating"
    READING = "reading"
    PACING = "pacing"
    FALL = "fall"
    AGGRESSIVE_MOVEMENT = "aggressive_movement"
    ATTACK_ON_PERSON = "attack_on_person"
    HEAD_BANGING = "head_banging"
    PROPERTY_DAMAGE = "property_damage"
    CHOKING_SIMULATION = "choking_simulation"
    VOMITING = "vomiting"
    LIGATURE_RISK = "ligature_risk"
    CUTTING_RISK = "cutting_risk"
    BLEEDING_VISIBLE = "bleeding_visible"
    SHARP_OBJECT_DETECTED = "sharp_object_detected"
    PROLONGED_INACTIVITY = "prolonged_inactivity"


class DatasetCategory(str, Enum):
    NORMAL = "normal"
    RISK = "risk"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
