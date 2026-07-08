from fastapi import APIRouter, File, UploadFile, status

from app.schemas.video import VideoUploadResponse
from app.services.video_service import save_uploaded_video


router = APIRouter(prefix="/videos")


@router.post(
    "/upload",
    response_model=VideoUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_video(file: UploadFile = File(...)) -> VideoUploadResponse:
    return save_uploaded_video(file)
