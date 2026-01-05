from pydantic import BaseModel
from typing import Optional

class BiologicalEffect(BaseModel):
    effectId: str
    name: str
    category: Optional[str] = None
