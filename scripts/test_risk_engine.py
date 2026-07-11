import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.pose_detector import PoseMovementSummary
from ml.inference.risk_engine import RiskSignalSnapshot, assess_risk


def main() -> int:
    examples = {
        "normal": RiskSignalSnapshot(action_group="normal_activity"),
        "dangerous_object": RiskSignalSnapshot(
            action_group="normal_activity",
            dangerous_objects=["scissors"],
        ),
        "clinical_object_cue": RiskSignalSnapshot(
            action_group="normal_activity",
            clinical_object_cues=["possible_ligature_cue"],
        ),
        "high_movement": RiskSignalSnapshot(
            action_group="normal_activity",
            pose_summary=PoseMovementSummary(
                frames_sampled=24,
                pose_frames=22,
                pose_coverage=0.92,
                mean_motion=0.08,
                max_motion=0.42,
                vertical_motion=0.40,
                horizontal_motion=0.35,
                posture_change=0.08,
            ),
        ),
        "low_movement": RiskSignalSnapshot(
            action_group="normal_activity",
            pose_summary=PoseMovementSummary(
                frames_sampled=24,
                pose_frames=24,
                pose_coverage=1.0,
                mean_motion=0.002,
                max_motion=0.004,
                vertical_motion=0.02,
                horizontal_motion=0.02,
                posture_change=0.01,
            ),
        ),
        "high_risk_group": RiskSignalSnapshot(action_group="violence_property_risk"),
    }

    for name, signals in examples.items():
        assessment = assess_risk(signals)
        print(
            f"{name}: group={assessment.risk_group} "
            f"level={assessment.risk_level} "
            f"alarm={assessment.alarm_required} "
            f"summary={assessment.observation_summary}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
