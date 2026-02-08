import torch
import numpy as np
from app.ml.world_model.plausibility_scorer import PlausibilityScorer

def test_scorer():
    scorer = PlausibilityScorer()
    
    # Mock Embeddings (Normalized for easier mental math)
    # A and B are distinct
    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([0.0, 1.0, 0.0])
    
    # C is identical to A
    emb_c = np.array([1.0, 0.0, 0.0])
    
    # D is somewhat similar to A
    emb_d = np.array([0.9, 0.1, 0.0])
    
    # Case 1: Diverse/Complementary Pair (A + B)
    # Cosine Sim should be 0.0 (Dot product)
    # Redundancy penalty should be 0.
    # Diversity Score should be 1.0.
    compounds_1 = [
        {'embedding': emb_a, 'risk_score': 0.0, 'name': 'A'},
        {'embedding': emb_b, 'risk_score': 0.0, 'name': 'B'}
    ]
    score_1, expl_1 = scorer.score_formulation(compounds_1)
    print(f"Case 1 (Diverse): Score={score_1:.4f}, Coherence={expl_1['coherence_metric']:.4f}")
    assert expl_1['redundancy_penalty'] == 0.0
    
    # Case 2: Highly Redundant Pair (A + C)
    # Cosine Sim should be 1.0.
    # Redundancy penalty should be (1.0 - 0.85) * 2 = 0.3.
    # Diversity Score = 0.7.
    compounds_2 = [
        {'embedding': emb_a, 'risk_score': 0.0, 'name': 'A'},
        {'embedding': emb_c, 'risk_score': 0.0, 'name': 'C'}
    ]
    score_2, expl_2 = scorer.score_formulation(compounds_2)
    print(f"Case 2 (Redundant): Score={score_2:.4f}, Coherence={expl_2['coherence_metric']:.4f}, Penalty={expl_2['redundancy_penalty']:.4f}")
    assert expl_2['redundancy_penalty'] > 0.1 
    
    # Case 3: Toxic Pair
    compounds_3 = [
        {'embedding': emb_a, 'risk_score': 0.8, 'name': 'A_Toxic'},
        {'embedding': emb_b, 'risk_score': 0.0, 'name': 'B'}
    ]
    score_3, expl_3 = scorer.score_formulation(compounds_3)
    print(f"Case 3 (Toxic): Score={score_3:.4f}, MaxRisk={expl_3['max_risk']:.4f}")
    assert score_3 < score_1 # Toxicity should lower the score
    
    print("All tests passed.")

if __name__ == "__main__":
    test_scorer()
