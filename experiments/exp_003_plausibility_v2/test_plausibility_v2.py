import sys
import os
import csv
import json
import random
sys.path.append(os.getcwd())

from backend.app.services.neo4j_service import Neo4jService
from backend.app.ml.world_model.plausibility_scorer import PlausibilityScorer

def main():
    print("Initializing Plausibility Experiment V2 (Logic Fixes)...")
    db = Neo4jService()
    scorer = PlausibilityScorer(db)
    
    # OUTPUT Setup
    exp_dir = 'experiments/exp_003_plausibility_v2'
    os.makedirs(exp_dir, exist_ok=True)
    csv_file = os.path.join(exp_dir, 'scores.csv')
    
    test_cases = [
        # Single (Baseline) - Should have 0 redundancy now
        {"name": "Baseline: Turmeric", "herbs": ["HERB_TURMERIC"], "type": "baseline"},
        
        # Known Synergistic
        {"name": "Turmeric + Pippali", "herbs": ["HERB_TURMERIC", "HERB_PIPPALI"], "type": "synergy"},
        
        # Redundant (Target Overlap + Chem Diff?) -> Should decrease penalty in V2 (Complementary?)
        # Or if we assume distinct chem, it should be high coherence, low redundancy.
        {"name": "Turmeric + Ginger", "herbs": ["HERB_TURMERIC", "HERB_GINGER"], "type": "redundancy_check"},
        
        # Broad Formulation
        {"name": "Immunity Triad (Amla, Guduchi, Ashwagandha)", 
         "herbs": ["HERB_AMLA", "HERB_GUDUCHI", "HERB_ASHWAGANDHA"], "type": "formulation"},
         
        # Random / Noise
        {"name": "Random Mix 1", "herbs": ["HERB_NEEM", "HERB_LICORICE"], "type": "random"},
        {"name": "Random Mix 2", "herbs": ["HERB_BRAHMI", "HERB_TULSI"], "type": "random"},
    ]
    
    # Save test cases
    with open(os.path.join(exp_dir, 'test_cases.json'), 'w') as f:
        json.dump(test_cases, f, indent=4)
        
    print(f"Running {len(test_cases)} Test Cases...")
    
    results = []
    
    for case in test_cases:
        score, explanation = scorer.score_combination(case['herbs'])
        
        # Extract components
        comps = explanation.get('components', {})
        
        res_row = {
            'name': case['name'],
            'type': case['type'],
            'herbs': "|".join(case['herbs']),
            'score': score,
            'coverage': comps.get('coverage', 0),
            'coherence': comps.get('coherence', 0),
            'redundancy_penalty': comps.get('redundancy_penalty', 0),
            'risk_penalty': comps.get('risk_penalty', 0),
            'interpretation': explanation.get('interpretation', '')
        }
        results.append(res_row)
        print(f"[{case['type'].upper()}] {case['name']}: {score:.3f} (Red: {res_row['redundancy_penalty']}, Coh: {res_row['coherence']})")
        
    # Validation Checks
    print("\n--- Sanity Checks V2 ---")
    
    # 1. Single Herb Redundancy
    turmeric = next((r for r in results if r['name'] == "Baseline: Turmeric"), None)
    if turmeric['redundancy_penalty'] == 0.0:
        print("PASS: Single herb redundancy is 0.0.")
    else:
        print(f"FAIL: Single herb redundancy is {turmeric['redundancy_penalty']}.")

    # 2. Turmeric+Ginger vs Turmeric+Pippali
    # Turmeric+Ginger (Overlap + Distinct Chem) should now ideally be high coherence, moderate redundancy penalty?
    # Wait, simple heuristic said ChemSim=0.2 if distinct names.
    # So Red = Coherence * 0.2.
    # Turmeric+Pippali (Less overlap?) -> Lower Coherence?
    # Let's see the scores.
    
    t_g = next((r for r in results if r['name'] == "Turmeric + Ginger"), None)
    t_p = next((r for r in results if r['name'] == "Turmeric + Pippali"), None)
    
    if t_g and t_p:
         print(f"Turmeric+Ginger: {t_g['score']:.3f} (Coh: {t_g['coherence']})")
         print(f"Turmeric+Pippali: {t_p['score']:.3f} (Coh: {t_p['coherence']})")
             
    # Save CSV
    keys = results[0].keys()
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Results saved to {csv_file}")
    
    scorer.close()
    db.close()

if __name__ == "__main__":
    main()
