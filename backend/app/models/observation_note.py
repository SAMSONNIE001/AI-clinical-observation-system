from pydantic import BaseModel
from datetime import datetime


class ObservationNote(BaseModel):
    patient_id: int
    generated_at: datetime
    note: str