from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class Position(BaseModel):
    x: float
    y: float

class NodeState(BaseModel):
    id: str
    type: str  # herb, compound, target, disease, effect, evidence, result
    position: Position
    data: Dict[str, Any] = {}
    
    # ReactFlow specific optional fields
    width: Optional[float] = None
    height: Optional[float] = None
    selected: Optional[bool] = None

class EdgeState(BaseModel):
    id: str
    source: str
    target: str
    label: Optional[str] = None

class Workspace(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    nodes: List[NodeState] = []
    edges: List[EdgeState] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
