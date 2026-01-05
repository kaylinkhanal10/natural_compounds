from pydantic import BaseModel
from typing import Optional, List
from .compound import Compound
from .effect import BiologicalEffect
from .evidence import Evidence

class HerbBase(BaseModel):
    herbId: str
    name: str
    scientificName: Optional[str] = None
    description: Optional[str] = None
    sanskritName: Optional[str] = None

class Herb(HerbBase):
    pass

class HerbDetail(Herb):
    compounds: List[Compound] = []
    effects: List[BiologicalEffect] = []
    evidence: List[Evidence] = []
