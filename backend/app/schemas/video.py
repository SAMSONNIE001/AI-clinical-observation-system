from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class VideoUploadResponse(BaseModel):
    id: str
    original_filename: str
    stored_filename: str
    content_type: str
    size_bytes: int
    storage_path: Path
    uploaded_at: datetime
