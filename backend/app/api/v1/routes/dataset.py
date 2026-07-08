from fastapi import APIRouter, status

from app.models.video_annotation import VideoAnnotation
from app.models.video_record import VideoRecord
from app.schemas.dataset import VideoAnnotationCreate, VideoRecordCreate
from app.services.dataset_service import (
    create_video_annotation,
    create_video_record,
    list_video_annotations,
    list_video_records,
)


router = APIRouter(prefix="/dataset")


@router.post(
    "/videos",
    response_model=VideoRecord,
    status_code=status.HTTP_201_CREATED,
)
def register_video_record(payload: VideoRecordCreate) -> VideoRecord:
    return create_video_record(payload)


@router.get("/videos", response_model=list[VideoRecord])
def get_video_records() -> list[VideoRecord]:
    return list_video_records()


@router.post(
    "/annotations",
    response_model=VideoAnnotation,
    status_code=status.HTTP_201_CREATED,
)
def register_video_annotation(payload: VideoAnnotationCreate) -> VideoAnnotation:
    return create_video_annotation(payload)


@router.get(
    "/annotations/{video_id}",
    response_model=list[VideoAnnotation],
)
def get_video_annotations(video_id: str) -> list[VideoAnnotation]:
    return list_video_annotations(video_id=video_id)
