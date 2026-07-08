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
    CHOKING_SIMULATION = "choking_simulation"
    PROLONGED_INACTIVITY = "prolonged_inactivity"


class DatasetCategory(str, Enum):
    NORMAL = "normal"
    RISK = "risk"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
