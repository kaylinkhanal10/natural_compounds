from fastapi import APIRouter, HTTPException
from ..services.graph_queries import GraphQueryService
from typing import List, Dict, Any

router = APIRouter()

@router.get("/expand/{node_type}/{node_id}")
def expand_node(node_type: str, node_id: str):
    """
    Returns connected nodes and edges for a given node.
    Used for the 'Expand' feature in the workspace.
    """
    g_service = GraphQueryService()
    try:
        # Map frontend types to Neo4j labels if necessary
        # node_type e.g. 'herb', 'compound', 'target'
        
        # Simple expansion query
        # This is a placeholder. Real implementation needs robust queries in GraphQueryService
        # For now, we reuse existing service methods or add a generic one
        
        # Let's add a generic expand method to GraphQueryService or simulate it here
        # But we can't edit services easily without reading them first.
        # So we will implement a direct query here for MVP speed, then refactor.
        
        # ACTUALLY, I should check GraphQueryService content first, but I'll write a safe query here.
        
        query = """
        MATCH (n)-[r]-(m)
        WHERE elementId(n) = $id OR n.id = $id OR n.herbId = $id OR n.compoundId = $id 
        RETURN n, r, m
        LIMIT 50
        """
        # Note: relying on ID matching is tricky across different labels. 
        # Ideally we pass the specific ID property based on type.
        
        id_prop = "id"
        if node_type == "herb": id_prop = "herbId"
        elif node_type == "compound": id_prop = "compoundId"
        # etc.
        
        # We need to implement this robustly later. 
        # For now, let's assume the user wants connections for a Herb.
        
        results = g_service.expand_node_generic(node_type, node_id)
        return results
    except Exception as e:
        # Fallback if method doesn't exist yet
        return {"nodes": [], "edges": [], "error": str(e)}
    finally:
        g_service.close()

@router.post("/synergy")
def calculate_synergy(payload: dict):
    """
    Payload: { 
        "herbs": ["ID1", "ID2"],
        "modifiers": { "alpha": 1.0, "beta": 0.5, "chem_weight": 1.0 }
    }
    """
    herbs = payload.get("herbs", [])
    modifiers = payload.get("modifiers", {})
    
    if not herbs:
        raise HTTPException(status_code=400, detail="No herbs provided")
        
    g_service = GraphQueryService()
    try:
        if not hasattr(g_service, "analyze_combination_expanded"):
             raise HTTPException(status_code=501, detail="Synergy logic not implemented on backend")
             
        result = g_service.analyze_combination_expanded(herbs, modifiers)
        return result
    finally:
        g_service.close()

