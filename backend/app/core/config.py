from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_name: str = "AI Clinical Observation System"
    version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"
    upload_dir: Path = Path("storage/uploads/videos")
    max_video_size_mb: int = 500


settings = Settings()
