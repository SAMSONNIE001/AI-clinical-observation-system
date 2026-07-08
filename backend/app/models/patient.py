from pydantic import BaseModel
from typing import Optional
from datetime import date


class Patient(BaseModel):
    id: int
    first_name: str
    last_name: str
    date_of_birth: date
    ward: Optional[str] = None
    observation_level: str