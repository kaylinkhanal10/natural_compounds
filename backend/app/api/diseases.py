from fastapi import APIRouter, HTTPException
from ..services.graph_queries import GraphQueryService

router = APIRouter()

@router.get("/")
def get_diseases():
    service = GraphQueryService()
    try:
        return service.get_diseases()
    finally:
        service.close()

@router.get("/{id}")
def get_disease_detail(id: str):
    service = GraphQueryService()
    try:
        data = service.get_disease_details(id)
        if not data:
            raise HTTPException(status_code=404, detail="Disease not found")
        return data
    finally:
        service.close()
