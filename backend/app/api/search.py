from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from ..services.neo4j_service import Neo4jService

router = APIRouter()

class SearchResult(BaseModel):
    id: str
    name: str # Display Name
    type: str # Herb, Disease, Target
    meta: Optional[str] = None # Scientific name, or other info
    count: Optional[int] = 0 # Number of child items (Compounds, Targets, etc.)

@router.get("/global", response_model=List[SearchResult])
def global_search(q: str = Query(..., min_length=2)):
    """
    Unified search for Herbs, Diseases, and Targets.
    Returns counts of associated items.
    """
    db = Neo4jService()
    results = []
    
    try:
        # 1. Search Herbs (Name, Scientific, Pinyin)
        q_herbs = """
        MATCH (h:Herb)
        WHERE toLower(h.name) CONTAINS toLower($q) 
           OR toLower(h.scientificName) CONTAINS toLower($q)
           OR (h.pinyin IS NOT NULL AND toLower(h.pinyin) CONTAINS toLower($q))
        OPTIONAL MATCH (h)-[:HAS_COMPOUND]->(c:Compound)
        WITH h, count(c) as c_count
        RETURN h.herbId as id, h.name as name, h.scientificName as meta, 'Herb' as type, c_count as count
        LIMIT 5
        """
        
        # 2. Search Diseases (Count Targets)
        q_diseases = """
        MATCH (d:Disease)
        WHERE toLower(d.name) CONTAINS toLower($q)
        OPTIONAL MATCH (p:Protein)-[:ASSOCIATED_WITH]->(d)
        WITH d, count(p) as p_count
        RETURN d.name as id, d.name as name, 'Disease' as type, p_count as count
        LIMIT 5
        """
        
        # 3. Search Targets (Count Compounds)
        q_targets = """
        MATCH (t:Protein)
        WHERE toLower(t.name) CONTAINS toLower($q)
        OPTIONAL MATCH (c:Compound)-[:TARGETS]->(t)
        WITH t, count(c) as c_count
        RETURN t.name as id, t.name as name, 'Target' as type, c_count as count
        LIMIT 5
        """
        
        with db.driver.session() as session:
            # Run in parallel or sequential - sequential is fine for now
            res_h = session.run(q_herbs, q=q.strip())
            results.extend([dict(r) for r in res_h])
            
            res_d = session.run(q_diseases, q=q.strip())
            results.extend([dict(r) for r in res_d])
            
            res_t = session.run(q_targets, q=q.strip())
            results.extend([dict(r) for r in res_t])
            
        return results
    finally:
        db.close()

@router.get("/context/{type}/{id}")
def get_context_compounds(type: str, id: str):
    """
    Get compounds associated with a specific entity (Herb, Disease, Target).
    """
    db = Neo4jService()
    try:
        query = ""
        params = {"id": id}
        
        if type == "Herb":
            # Re-use existing logic logic or call internal service? 
            # Let's write a direct optimized query for the "Inventory" view
            # We need standard compound props + source tagging
            query = """
            MATCH (h:Herb {herbId: $id})
            OPTIONAL MATCH (h)-[:HAS_COMPOUND]->(c:Compound)
            RETURN c.compoundId as compoundId, c.name as name, 
                   c.inchikey as inchikey, c.smiles as smiles, c.mw as mw, c.logp as logp, 
                   c.tpsa as tpsa, h.name as sourceHerb
            """
            
        elif type == "Disease":
            # Compounds that treat this disease (via Targets or direct?)
            # Valid relationships: (Compound)-[:TARGETS]->(Protein)-[:ASSOCIATED_WITH]->(Disease)
            # OR (Compound)-[:TREATS]->(Disease) if explicit
            query = """
            MATCH (d:Disease {name: $id})
            MATCH (p:Protein)-[:ASSOCIATED_WITH]->(d)
            MATCH (c:Compound)-[:TARGETS]->(p)
            RETURN DISTINCT c.compoundId as compoundId, c.name as name, 
                   c.inchikey as inchikey, c.smiles as smiles, c.mw as mw, c.logp as logp, 
                   c.tpsa as tpsa, 'Targeting: ' + p.name as sourceHerb 
            LIMIT 50
            """
            
        elif type == "Target":
            query = """
            MATCH (t:Protein {name: $id})
            MATCH (c:Compound)-[:TARGETS]->(t)
            RETURN DISTINCT c.compoundId as compoundId, c.name as name, 
                   c.inchikey as inchikey, c.smiles as smiles, c.mw as mw, c.logp as logp, 
                   c.tpsa as tpsa, 'Target: ' + t.name as sourceHerb
            LIMIT 50
            """
            
        else:
            raise HTTPException(status_code=400, detail="Invalid type")
            
        with db.driver.session() as session:
            res = session.run(query, params)
            return [dict(r) for r in res]
            
    finally:
        db.close()

@router.get("/context/{type}/{id}/children", response_model=List[SearchResult])
def get_context_children(type: str, id: str):
    """
    Get child entities for hierarchical navigation.
    e.g., Disease -> Targets
    """
    db = Neo4jService()
    try:
        results = []
        if type == "Disease":
            # Disease -> Associated Proteins (Targets)
            # Count how many compounds target each protein
            query = """
            MATCH (d:Disease {name: $id})<-[:ASSOCIATED_WITH]-(t:Protein)
            OPTIONAL MATCH (c:Compound)-[:TARGETS]->(t)
            WITH t, count(c) as c_count
            RETURN t.name as id, t.name as name, 'Target' as type, 'Associated Target' as meta, c_count as count
            ORDER BY c_count DESC
            LIMIT 50
            """
            with db.driver.session() as session:
                res = session.run(query, id=id)
                results = [dict(r) for r in res]
                
        # Future: Herb -> Chemical Classes?
        # Future: Pathway -> Targets?
        
        return results
    finally:
        db.close()
