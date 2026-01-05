from fastapi import APIRouter, HTTPException
from ..services.graph_queries import GraphQueryService

router = APIRouter()

@router.get("/")
def get_formulas():
    service = GraphQueryService()
    try:
        return service.get_formulas()
    finally:
        service.close()

@router.get("/{name}")
def get_formula_detail(name: str):
    service = GraphQueryService()
    try:
        data = service.get_formula_details(name)
        if not data:
            raise HTTPException(status_code=404, detail="Formula not found")
        return data
    finally:
        service.close()
