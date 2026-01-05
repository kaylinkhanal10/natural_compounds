from pydantic import BaseModel
from typing import Optional

class Compound(BaseModel):
    compoundId: str
    name: str
    chemicalClass: Optional[str] = None
    inchikey: Optional[str] = None
    mw: Optional[float] = None
    logp: Optional[float] = None
    tpsa: Optional[float] = None
    hba: Optional[int] = None
    hbd: Optional[int] = None
    rotb: Optional[int] = None
    nring: Optional[int] = None
    formula: Optional[str] = None
    property_source: Optional[str] = None
    confidence: Optional[str] = None
