import sys
import os
import csv
import json
sys.path.append(os.getcwd())

from backend.app.services.neo4j_service import Neo4jService
from backend.app.ml.world_model.plausibility_scorer import PlausibilityScorer

def main():
    print("Initializing Compound-Level Plausibility Experiment (EXP-003)...")
    db = Neo4jService()
    scorer = PlausibilityScorer(db)
    
    # OUTPUT Setup
    exp_dir = 'experiments/exp_003_compound_logic'
    os.makedirs(exp_dir, exist_ok=True)
    csv_file = os.path.join(exp_dir, 'scores.csv')
    
    test_cases = [
        # Single (Baseline)
        {"name": "Baseline: Turmeric", "herbs": ["HERB_TURMERIC"], "type": "baseline"},
        
        # Known Synergistic (Classic Pair)
        {"name": "Turmeric + Pippali", "herbs": ["HERB_TURMERIC", "HERB_PIPPALI"], "type": "synergy"},
        
        # Formulation
        {"name": "Immunity Triad (Amla, Guduchi, Ashwagandha)", 
         "herbs": ["HERB_AMLA", "HERB_GUDUCHI", "HERB_ASHWAGANDHA"], "type": "formulation"},
         
         # Random / Noise
        {"name": "Random Mix 1", "herbs": ["HERB_NEEM", "HERB_LICORICE"], "type": "random"},
    ]
    
    print(f"Running {len(test_cases)} Test Cases...")
    
    results = []
    
    for case in test_cases:
        score, explanation = scorer.score_combination(case['herbs'])
        
        # Extract components
        comps = explanation.get('components', {})
        details = explanation.get('details', {})
        
        res_row = {
            'name': case['name'],
            'type': case['type'],
            'herbs': "|".join(case['herbs']),
            'score': score,
            'coverage': comps.get('coverage', 0),
            'coherence': comps.get('coherence', 0),
            'redundancy_penalty': comps.get('redundancy_penalty', 0),
            'compounds_used': details.get('compound_count', 0),
            'targets_hit': details.get('target_count', 0),
            'interpretation': explanation.get('interpretation', '')
        }
        results.append(res_row)
        print(f"[{case['type'].upper()}] {case['name']}: {score:.3f} (Compounds: {res_row['compounds_used']}, Targets: {res_row['targets_hit']})")
        
    # Save CSV
    keys = results[0].keys()
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Results saved to {csv_file}")
    
    # ABLATION Check (Turmeric + Pippali vs Turmeric)
    turm = next((r for r in results if r['name'] == "Baseline: Turmeric"), None)
    turm_pip = next((r for r in results if r['name'] == "Turmeric + Pippali"), None)
    
    if turm and turm_pip:
        print("\n--- ABLATION CHECK ---")
        print(f"Turmeric: {turm['score']:.3f}")
        print(f"Turmeric + Pippali: {turm_pip['score']:.3f}")
        if turm_pip['score'] > turm['score']:
            print("PASS: Combination > Single Herb (Synergy detected or coverage boost)")
        else:
            print("INFO: Combination <= Single Herb (Maybe redundancy or dilution?)")

    scorer.close()
    db.close()

if __name__ == "__main__":
    main()
