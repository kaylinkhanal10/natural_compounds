from fastapi import APIRouter, HTTPException, Body
from typing import List, Optional
import json
import os
from ..models.workspace import Workspace

router = APIRouter()

# Simple file-based persistence for MVP
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
WORKSPACE_FILE = os.path.join(DATA_DIR, "workspaces.json")

def load_workspaces():
    if not os.path.exists(WORKSPACE_FILE):
        return {}
    try:
        with open(WORKSPACE_FILE, "r") as f:
            data = json.load(f)
            return {k: Workspace(**v) for k, v in data.items()}
    except Exception:
        return {}

def save_workspaces(workspaces):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(WORKSPACE_FILE, "w") as f:
        json.dump({k: v.dict(exclude_none=True) for k, v in workspaces.items()}, f, default=str)

@router.get("/", response_model=List[Workspace])
def get_workspaces():
    workspaces = load_workspaces()
    # Return sorted by updated_at desc
    return sorted(workspaces.values(), key=lambda w: w.updated_at, reverse=True)

@router.post("/", response_model=Workspace)
def create_workspace(name: str = Body(..., embed=True), description: Optional[str] = Body(None, embed=True)):
    workspaces = load_workspaces()
    new_workspace = Workspace(name=name, description=description)
    workspaces[new_workspace.id] = new_workspace
    save_workspaces(workspaces)
    return new_workspace

@router.get("/{workspace_id}", response_model=Workspace)
def get_workspace(workspace_id: str):
    workspaces = load_workspaces()
    if workspace_id not in workspaces:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspaces[workspace_id]

@router.put("/{workspace_id}", response_model=Workspace)
def update_workspace(workspace_id: str, workspace: Workspace):
    workspaces = load_workspaces()
    if workspace_id not in workspaces:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Ensure ID match
    if workspace.id != workspace_id:
        raise HTTPException(status_code=400, detail="Workspace ID mismatch")
        
    workspaces[workspace_id] = workspace
    save_workspaces(workspaces)
    return workspace

@router.delete("/{workspace_id}")
def delete_workspace(workspace_id: str):
    workspaces = load_workspaces()
    if workspace_id in workspaces:
        del workspaces[workspace_id]
        save_workspaces(workspaces)
    return {"status": "success"}
