import sys
import os
import csv
import json
import random
sys.path.append(os.getcwd())

from backend.app.services.neo4j_service import Neo4jService
from backend.app.ml.world_model.plausibility_scorer import PlausibilityScorer

def main():
    print("Initializing Plausibility Experiment...")
    db = Neo4jService()
    scorer = PlausibilityScorer(db)
    
    # OUTPUT Setup
    exp_dir = 'experiments/exp_002_plausibility'
    os.makedirs(exp_dir, exist_ok=True)
    csv_file = os.path.join(exp_dir, 'scores.csv')
    
    # 1. Define Test Cases
    # Based on available herbs in MVP (Amla, Guduchi, etc.)
    # Known Synergistic (Hypothetical for MVP): Turmeric + Pippali (Bioavailability)
    # Redundant: Turmeric + Ginger (Both anti-inflammatory, similar chemistry?)
    # Random: 3 random herbs
    
    # We need valid IDs. 
    # From previous query: HERB_AMLA, HERB_GUDUCHI, HERB_ASHWAGANDHA, HERB_TURMERIC, 
    # HERB_GINGER, HERB_TULSI, HERB_NEEM, HERB_BRAHMI, HERB_PIPPALI, HERB_LICORICE
    
    test_cases = [
        # Single (Baseline)
        {"name": "Baseline: Turmeric", "herbs": ["HERB_TURMERIC"], "type": "baseline"},
        
        # Known Synergistic (Classic Pair)
        {"name": "Turmeric + Pippali", "herbs": ["HERB_TURMERIC", "HERB_PIPPALI"], "type": "synergy"},
        
        # Redundant (Similar profile)
        {"name": "Turmeric + Ginger", "herbs": ["HERB_TURMERIC", "HERB_GINGER"], "type": "redundancy_check"},
        
        # Broad Formulation (Triad)
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
        print(f"[{case['type'].upper()}] {case['name']}: {score:.3f} (Redundancy: {res_row['redundancy_penalty']})")
        
    # Validation Checks
    # 1. Does Synergy (Turmeric+Pippali) score higher than Random?
    # 2. Does Redundancy Penalty trigger for Turmeric+Ginger?
    
    turmeric_pippali = next((r for r in results if r['name'] == "Turmeric + Pippali"), None)
    turmeric_ginger = next((r for r in results if r['name'] == "Turmeric + Ginger"), None)
    
    print("\n--- Sanity Checks ---")
    if turmeric_pippali and turmeric_ginger:
        diff = turmeric_pippali['score'] - turmeric_ginger['score']
        print(f"Score Diff (Syn - Red): {diff:.3f}")
        if diff > -0.1: # Allow slight margin, ideally positive
             print("PASS: Synergy pair comparable or better than redundancy pair.")
        else:
             print("WARN: Redundant pair scored higher.")
             
    # Save CSV
    keys = results[0].keys()
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Results saved to {csv_file}")
    
    # ABLATION EXPERIMENT
    # Take "Immunity Triad" and remove one herb at a time
    print("\nRunning Ablation on Immunity Triad...")
    triad = ["HERB_AMLA", "HERB_GUDUCHI", "HERB_ASHWAGANDHA"]
    base_score, _ = scorer.score_combination(triad)
    print(f"Base Score (Triad): {base_score:.3f}")
    
    ablation_results = []
    for h in triad:
        subset = [x for x in triad if x != h]
        sub_score, _ = scorer.score_combination(subset)
        delta = base_score - sub_score
        print(f"Removed {h}: New Score {sub_score:.3f} (Delta: {delta:.3f})")
        ablation_results.append({
            'removed': h,
            'new_score': sub_score,
            'delta': delta
        })
        
    # Save Ablation
    with open(os.path.join(exp_dir, 'ablation_results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['removed', 'new_score', 'delta'])
        writer.writeheader()
        writer.writerows(ablation_results)
        
    scorer.close()
    db.close()

if __name__ == "__main__":
    main()
