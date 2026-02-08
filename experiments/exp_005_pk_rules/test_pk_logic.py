import sys
import os
import csv
import json
sys.path.append(os.getcwd())

from backend.app.services.neo4j_service import Neo4jService
from backend.app.ml.world_model.plausibility_scorer import PlausibilityScorer

def main():
    print("Initializing EXP-005: Pharmacokinetic Rules...")
    db = Neo4jService()
    scorer = PlausibilityScorer(db)
    
    exp_dir = 'experiments/exp_005_pk_rules'
    os.makedirs(exp_dir, exist_ok=True)
    csv_file = os.path.join(exp_dir, 'scores.csv')
    
    # Mock Data for Pure Compound Scoring
    compounds_db = {
        "CURCUMIN": {
            "name": "Curcumin",
            "smiles": "COC1=C(C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC)O",
            "targets": ["PTGS2", "TNF", "NFKB1"] # Anti-inflammatory
        },
        "PIPERINE": {
             "name": "Piperine",
             "smiles": "C1CCN(CC1)C(=O)/C=C/C=C/C2=CC3=C(C=C2)OCO3",
             "targets": ["CYP3A4", "ABCB1", "TRPV1"] # PK Enhancer Targets!
        },
        "NON_ENHANCER": {
            "name": "Dummy_Chemical",
            "smiles": "CCCCCC",
            "targets": ["TRPV1"] # No PK targets, No Overlap
        }
    }
    
    test_cases = [
        {
            "name": "Curcumin + Piperine (PK Synergy)",
            "groups": {
                "Curcumin": [compounds_db["CURCUMIN"]],
                "Piperine": [compounds_db["PIPERINE"]]
            },
            "type": "pk_synergy"
        },
        {
            "name": "Curcumin + Dummy (Control)",
            "groups": {
                "Curcumin": [compounds_db["CURCUMIN"]],
                "Dummy": [compounds_db["NON_ENHANCER"]]
            },
            "type": "control" 
        }
    ]
    
    print(f"Running {len(test_cases)} Tests...")
    
    results = []
    
    for case in test_cases:
        score, explanation = scorer.score_direct(case['groups'])
        
        comps = explanation.get('components', {})
        details = explanation.get('details', {})
        interactions = details.get('interactions', [])
        
        pk_boost = 0
        if interactions:
            pk_boost = interactions[0].get('pk_boost', 0)
        
        res_row = {
            'name': case['name'],
            'type': case['type'],
            'score': score,
            'coherence': comps.get('coherence', 0),
            'pk_boost': pk_boost,
            'interpretation': explanation.get('interpretation', '')
        }
        results.append(res_row)
        print(f"[{case['type'].upper()}] {case['name']}: {score:.3f} (Boost: {pk_boost})")
        
    # Save CSV
    keys = results[0].keys()
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Results saved to {csv_file}")
    
    # Assert
    pk_score = next(r['score'] for r in results if r['type'] == 'pk_synergy')
    ctrl_score = next(r['score'] for r in results if r['type'] == 'control')
    
    if pk_score > ctrl_score:
        print("\nSUCCESS: PK Enhancer Boosted the score significantly!")
    else:
        print("\nFAIL: PK Enhancer did not boost score.")

    scorer.close()
    db.close()

if __name__ == "__main__":
    main()
