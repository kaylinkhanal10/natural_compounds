from pydantic import BaseModel
from typing import Optional

class Evidence(BaseModel):
    evidenceId: str
    title: str
    url: Optional[str] = None
    doi: Optional[str] = None
    confidence: Optional[str] = None
