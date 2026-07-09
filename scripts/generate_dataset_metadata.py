import csv
import json
from pathlib import Path

import cv2


RAW_DATASET_DIR = Path("dataset/raw")
METADATA_CSV = Path("dataset/metadata.csv")
TRAINING_MANIFEST = Path("dataset/training_manifest.jsonl")

NORMAL_BEHAVIOURS = {
    "sleeping",
    "sitting",
    "standing",
    "walking",
    "eating",
    "reading",
}

SCENARIO_NAMES = {
    "aggressive_movement": "agitated_arm_movement_simulation",
    "choking_simulation": "safe_choking_gesture_simulation",
    "eating": "seated_eating_simulation",
    "fall": "staged_low_risk_fall",
    "pacing": "repeated_walk_pattern",
    "prolonged_inactivity": "stillness_in_place",
    "reading": "seated_reading",
    "sitting": "seated_idle",
    "sleeping": "lying_still_on_bed",
    "standing": "standing_idle",
    "vomiting": "safe_vomiting_gesture_simulation",
    "walking": "walking_across_room",
}

NOTES = {
    "choking_simulation": "safe simulation only; no real airway obstruction",
    "fall": "staged safely only",
    "vomiting": "safe simulation only; no real vomiting",
}

FIELDNAMES = [
    "video_id",
    "filename",
    "relative_path",
    "behaviour_type",
    "category",
    "scenario_name",
    "duration_seconds",
    "environment",
    "camera_angle",
    "recorded_by",
    "recorded_at",
    "notes",
]


def main() -> None:
    records = []
    for video_path in sorted(RAW_DATASET_DIR.glob("*/*.mp4")):
        behaviour = video_path.parent.name
        expected_prefix = f"{behaviour}_"
        if not video_path.name.startswith(expected_prefix):
            raise ValueError(f"Unexpected filename for folder: {video_path}")

        video_id = video_path.stem
        duration_seconds = _video_duration_seconds(video_path)
        category = "normal" if behaviour in NORMAL_BEHAVIOURS else "risk"

        records.append(
            {
                "video_id": video_id,
                "filename": video_path.name,
                "relative_path": video_path.as_posix(),
                "behaviour_type": behaviour,
                "category": category,
                "scenario_name": SCENARIO_NAMES.get(behaviour, behaviour),
                "duration_seconds": f"{duration_seconds:.2f}",
                "environment": "simulated_room",
                "camera_angle": "mixed_angles",
                "recorded_by": "",
                "recorded_at": "",
                "notes": NOTES.get(behaviour, "self-simulated dataset clip"),
            }
        )

    if not records:
        raise ValueError(f"No .mp4 files found under {RAW_DATASET_DIR}")

    _write_metadata_csv(records)
    _write_training_manifest(records)
    print(f"Wrote {len(records)} records to {METADATA_CSV}")
    print(f"Wrote {len(records)} records to {TRAINING_MANIFEST}")


def _video_duration_seconds(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()

    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Could not read duration metadata: {video_path}")

    return frame_count / fps


def _write_metadata_csv(records: list[dict[str, str]]) -> None:
    with METADATA_CSV.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def _write_training_manifest(records: list[dict[str, str]]) -> None:
    with TRAINING_MANIFEST.open("w", encoding="utf-8") as output_file:
        for record in records:
            manifest_item = {
                "video_id": record["video_id"],
                "filename": record["filename"],
                "relative_path": record["relative_path"],
                "label": record["behaviour_type"],
                "category": record["category"],
                "scenario_name": record["scenario_name"],
                "duration_seconds": float(record["duration_seconds"]),
                "environment": record["environment"],
                "camera_angle": record["camera_angle"],
            }
            output_file.write(json.dumps(manifest_item) + "\n")


if __name__ == "__main__":
    main()
