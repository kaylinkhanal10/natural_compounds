from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ..services.reasoning_engine import ReasoningEngine

router = APIRouter()

class CombinationRequest(BaseModel):
    herbs: List[str]
    compounds: List[str] = []
    probiotics: List[str] = []
    minerals: List[str] = []

@router.post("/")
def combine_herbs(request: CombinationRequest):
    engine = ReasoningEngine()
    try:
        # Currently only supporting herbs logic for MVP
        if not request.herbs:
             raise HTTPException(status_code=400, detail="At least one herb must be selected.")
        
        result = engine.analyze_combination(request.herbs)
        
     
        # Add extended analysis
        from ..services.graph_queries import GraphQueryService
        g_service = GraphQueryService()
        try:
             extended = g_service.analyze_combination_expanded(request.herbs)
             
             # Enrich with Knowledge Dictionary (for full names)
             from ..services.target_knowledge import TargetKnowledgeService
             tk_service = TargetKnowledgeService()
             extended['target_dictionary'] = tk_service.knowledge_base
             
             result['extended'] = extended
             
             # Feasibility Logic
             feasibility_issues = []
             good_properties = []
             if 'chemical_feasibility' in extended:
                 for c in extended['chemical_feasibility']:
                     # Check rules: MW > 600, TPSA > 140
                     issues = []
                     if c['mw'] and c['mw'] > 600:
                         issues.append(f"MW {c['mw']} > 600")
                     if c['tpsa'] and c['tpsa'] > 140:
                         issues.append(f"TPSA {c['tpsa']} > 140")
                     
                     if issues:
                         feasibility_issues.append(f"{c['name']} ({', '.join(issues)})")
                     elif c['logp'] and 1 < c['logp'] < 5:
                         good_properties.append(f"{c['name']} (LogP {c['logp']})")
             
             # Append to explanation
             feasibility_text = "\n\nChemical Feasibility Analysis:\n"
             if feasibility_issues:
                 feasibility_text += f"Caution: Some compounds show poor oral bioavailability traits: {', '.join(feasibility_issues[:3])}."
                 if len(feasibility_issues) > 3: feasibility_text += "..."
             else:
                 feasibility_text += "Key driver compounds exhibit favorable drug-like properties."
                 
             if good_properties:
                 feasibility_text += f"\nComplementary Profiles: {good_properties[0]} complements polar constituents."
                 
             result['explanation'] += feasibility_text
             
        except Exception as e:
             # Log error
             with open("backend_error.log", "w") as f:
                 f.write(f"Extended analysis failed: {str(e)}")
             print(f"Extended analysis failed: {e}")
        finally:
             g_service.close()
             
        return result
    finally:
        engine.close()
