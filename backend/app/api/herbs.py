from fastapi import APIRouter, HTTPException
from ..services.neo4j_service import Neo4jService
from ..models.herb import HerbDetail

router = APIRouter()

@router.get("/", response_model=list)
def get_herbs():
    db = Neo4jService()
    try:
        herbs = db.get_all_herbs()
        return herbs
    finally:
        db.close()

@router.get("/{herb_id}", response_model=None)
def get_herb_detail(herb_id: str):
    # Try expanded first
    from ..services.graph_queries import GraphQueryService
    g_service = GraphQueryService()
    expanded = g_service.get_herb_details_expanded(herb_id)
    g_service.close()
    
    if expanded:
        return expanded
        
    # Fallback to old service if no extended data
    db = Neo4jService()
    try:
        herb = db.get_herb_details(herb_id)
        if not herb:
            raise HTTPException(status_code=404, detail="Herb not found")
        return herb
    finally:
        db.close()
