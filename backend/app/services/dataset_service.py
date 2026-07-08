import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, status

from app.core.config import settings
from app.models.video_annotation import VideoAnnotation
from app.models.video_record import VideoRecord
from app.schemas.dataset import (
    DatasetExportResponse,
    VideoAnnotationCreate,
    VideoRecordCreate,
)


VIDEO_RECORDS_FILE = "video_records.json"
VIDEO_ANNOTATIONS_FILE = "video_annotations.json"
TRAINING_MANIFEST_FILE = "training_manifest.jsonl"


def create_video_record(payload: VideoRecordCreate) -> VideoRecord:
    records = list_video_records()
    if any(record.video_id == payload.video_id for record in records):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A video record with this video_id already exists.",
        )

    record = VideoRecord(
        id=str(uuid4()),
        created_at=datetime.now(timezone.utc),
        **payload.model_dump(),
    )
    records.append(record)
    _write_items(_metadata_path(VIDEO_RECORDS_FILE), records)
    return record


def list_video_records() -> list[VideoRecord]:
    return [
        VideoRecord.model_validate(item)
        for item in _read_items(_metadata_path(VIDEO_RECORDS_FILE))
    ]


def create_video_annotation(payload: VideoAnnotationCreate) -> VideoAnnotation:
    if not any(record.video_id == payload.video_id for record in list_video_records()):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video record not found for this annotation.",
        )

    annotation = VideoAnnotation(
        id=str(uuid4()),
        created_at=datetime.now(timezone.utc),
        **payload.model_dump(),
    )
    annotations = list_video_annotations()
    annotations.append(annotation)
    _write_items(_metadata_path(VIDEO_ANNOTATIONS_FILE), annotations)
    return annotation


def list_video_annotations(video_id: str | None = None) -> list[VideoAnnotation]:
    annotations = [
        VideoAnnotation.model_validate(item)
        for item in _read_items(_metadata_path(VIDEO_ANNOTATIONS_FILE))
    ]
    if video_id is None:
        return annotations
    return [annotation for annotation in annotations if annotation.video_id == video_id]


def export_training_manifest() -> DatasetExportResponse:
    records = list_video_records()
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No video records available to export.",
        )

    export_id = str(uuid4())
    export_dir = settings.dataset_export_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / f"{export_id}_{TRAINING_MANIFEST_FILE}"

    with export_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            manifest_item = {
                "video_id": record.video_id,
                "filename": record.filename,
                "label": record.behaviour_type.value,
                "category": record.category.value,
                "scenario_name": record.scenario_name,
                "duration_seconds": record.duration_seconds,
                "environment": record.environment,
                "camera_angle": record.camera_angle,
            }
            output_file.write(json.dumps(manifest_item) + "\n")

    return DatasetExportResponse(
        export_id=export_id,
        format="jsonl",
        record_count=len(records),
        export_path=export_path,
        created_at=datetime.now(timezone.utc),
    )


def _metadata_path(filename: str) -> Path:
    settings.dataset_metadata_dir.mkdir(parents=True, exist_ok=True)
    return settings.dataset_metadata_dir / filename


def _read_items(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_items(path: Path, items: list[VideoRecord] | list[VideoAnnotation]) -> None:
    serialised = [item.model_dump(mode="json") for item in items]
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(serialised, output_file, indent=2)
