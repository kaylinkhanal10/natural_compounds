import sys
import os
import csv
import json
sys.path.append(os.getcwd())

from backend.app.services.neo4j_service import Neo4jService
from backend.app.ml.world_model.plausibility_scorer import PlausibilityScorer

def main():
    print("Initializing EXP-004: Pure Compound Interactions...")
    # Initialize scorer (Chemical Model will load)
    db = Neo4jService()
    scorer = PlausibilityScorer(db)
    
    # OUTPUT Setup
    exp_dir = 'experiments/exp_004_pure_compounds'
    os.makedirs(exp_dir, exist_ok=True)
    csv_file = os.path.join(exp_dir, 'scores.csv')
    
    # Define Pure Compounds (Mock Targets for now, or real if we had them handy)
    # We will simulate "Curcumin" and "Piperine" behavior
    # Curcumin: Anti-inflammatory (Targets: COX-2, TNF-a, NF-kB)
    # Piperine: CYP450 inhibitor (Targets: CYP3A4, P-gp) -> Distinct targets from Curcumin
    
    # Note: Plausibility V2 checks Intersect/Union.
    # If Targets are distinct (Bio Sim = 0), Coherence will be 0.
    # This confirms the previous finding.
    
    # Let's also test a "Positive Control" for Coherence:
    # Curcumin + Quercetin (Both hit Anti-inflammatory pathways? Overlap expected)
    
    compounds_db = {
        "CURCUMIN": {
            "name": "Curcumin",
            "smiles": "COC1=C(C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC)O",
            "targets": ["PTGS2", "TNF", "NFKB1", "IL6", "MAPK1"]
        },
        "PIPERINE": {
             "name": "Piperine",
             "smiles": "C1CCN(CC1)C(=O)/C=C/C=C/C2=CC3=C(C=C2)OCO3",
             "targets": ["CYP3A4", "ABCB1", "TRPV1"]
        },
        "QUERCETIN": {
            "name": "Quercetin",
            "smiles": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O",
            "targets": ["PTGS2", "TNF", "IL6", "MAPK1", "PIK3CA"] # High overlap with Curcumin
        },
        "DUMMY_REDUNDANT": {
            "name": "Curcumin_Analog",
            "smiles": "COC1=C(C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC)O", # Same SMILES 
            "targets": ["PTGS2", "TNF", "NFKB1", "IL6", "MAPK1"] # Same Targets
        }
    }
    
    test_cases = [
        {
            "name": "Curcumin (Baseline)",
            "groups": {
                "Curcumin": [compounds_db["CURCUMIN"]]
            },
            "type": "baseline"
        },
        {
            "name": "Curcumin + Piperine (Synergy)",
            "groups": {
                "Curcumin": [compounds_db["CURCUMIN"]],
                "Piperine": [compounds_db["PIPERINE"]]
            },
            "type": "synergy_pk" # Expected Low Coherence (PD), Needs PK rule
        },
        {
            "name": "Curcumin + Quercetin (Coherence)",
            "groups": {
                "Curcumin": [compounds_db["CURCUMIN"]],
                "Quercetin": [compounds_db["QUERCETIN"]]
            },
            "type": "synergy_pd" # Expected High Coherence
        },
        {
            "name": "Curcumin + Analog (Redundancy)",
            "groups": {
                "Curcumin": [compounds_db["CURCUMIN"]],
                "Analog": [compounds_db["DUMMY_REDUNDANT"]]
            },
            "type": "redundancy" # Expected High Coherence + High Chem Sim -> Penalty
        }
    ]
    
    print(f"Running {len(test_cases)} Pure Compound Cases...")
    
    results = []
    
    for case in test_cases:
        score, explanation = scorer.score_direct(case['groups'])
        
        comps = explanation.get('components', {})
        
        res_row = {
            'name': case['name'],
            'type': case['type'],
            'score': score,
            'coherence': comps.get('coherence', 0),
            'redundancy': comps.get('redundancy_penalty', 0),
            'interpretation': explanation.get('interpretation', '')
        }
        results.append(res_row)
        print(f"[{case['type'].upper()}] {case['name']}: {score:.3f} (Coh: {res_row['coherence']}, Red: {res_row['redundancy']})")
        
    # Save CSV
    keys = results[0].keys()
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    scorer.close()
    db.close()

if __name__ == "__main__":
    main()
