from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    project_name: str = "AI Clinical Observation System"
    version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"


settings = Settings()
