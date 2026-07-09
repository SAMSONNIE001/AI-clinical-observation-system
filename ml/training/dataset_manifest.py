import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingManifestItem:
    video_id: str
    filename: str
    label: str
    category: str
    scenario_name: str
    duration_seconds: float
    environment: str
    camera_angle: str | None = None
    relative_path: str | None = None


def load_training_manifest(path: Path) -> list[TrainingManifestItem]:
    with path.open("r", encoding="utf-8") as input_file:
        return [
            TrainingManifestItem(**json.loads(line))
            for line in input_file
            if line.strip()
        ]
