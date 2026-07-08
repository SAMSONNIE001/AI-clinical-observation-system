from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.video import VideoUploadResponse


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_VIDEO_CONTENT_TYPES = {
    "video/mp4",
    "video/x-msvideo",
    "video/quicktime",
    "video/x-matroska",
}


def save_uploaded_video(file: UploadFile) -> VideoUploadResponse:
    original_filename = file.filename or ""
    suffix = Path(original_filename).suffix.lower()

    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported video file type.",
        )

    if file.content_type not in ALLOWED_VIDEO_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported video content type.",
        )

    video_id = str(uuid4())
    stored_filename = f"{video_id}{suffix}"
    upload_dir = settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    storage_path = upload_dir / stored_filename

    size_bytes = 0
    max_size_bytes = settings.max_video_size_mb * 1024 * 1024

    with storage_path.open("wb") as output:
        while chunk := file.file.read(1024 * 1024):
            size_bytes += len(chunk)
            if size_bytes > max_size_bytes:
                output.close()
                storage_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Uploaded video exceeds the maximum allowed size.",
                )
            output.write(chunk)

    return VideoUploadResponse(
        id=video_id,
        original_filename=original_filename,
        stored_filename=stored_filename,
        content_type=file.content_type or "application/octet-stream",
        size_bytes=size_bytes,
        storage_path=storage_path,
        uploaded_at=datetime.now(timezone.utc),
    )
