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
    FIGHTING = "fighting"
    ATTACK_ON_PERSON = "attack_on_person"
    HEAD_BANGING = "head_banging"
    PROPERTY_DAMAGE = "property_damage"
    CHOKING_SIMULATION = "choking_simulation"
    LIGATURE_RISK = "ligature_risk"
    CUTTING_RISK = "cutting_risk"
    BLEEDING_VISIBLE = "bleeding_visible"
    BLOOD_VISIBLE = "blood_visible"
    SHARP_OBJECT_DETECTED = "sharp_object_detected"
    PROLONGED_INACTIVITY = "prolonged_inactivity"


class DatasetCategory(str, Enum):
    NORMAL = "normal"
    RISK = "risk"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
