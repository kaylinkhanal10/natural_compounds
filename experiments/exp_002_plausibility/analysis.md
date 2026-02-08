# Experiment 002: Plausibility Scorer Sanity Checks
**Date**: 2026-02-06
**Status**: Completed (with Findings)

## Objective
Verify the Boolean/Heuristic logic of the `PlausibilityScorer` using the Neo4j Knowledge Graph.
Focus Areas: Coherence (overlap), Redundancy (penalty), and Ablation sensitivity.

## Key Findings

### 1. Redundancy Logic Failure
**Observation**: The redundancy penalty failed to differentiate distinct pairs from overlapping pairs effectively.
- **Turmeric + Ginger** (High Overlap expected): Score **0.794** (Penalty: 0.0) -> Scored *higher* than synergistic pair.
- **Turmeric + Pippali** (Synergistic): Score **0.511** (Penalty: 0.0).
- **Cause**: The pairwise Jaccard index for Turmeric+Ginger did not exceed the strict `0.7` threshold required to trigger the penalty, likely due to sparse or incomplete target data in the current graph.

### 2. Single Herb Artifact
**Observation**: Single herbs (Baseline) received a massive redundancy penalty (0.6).
- **Cause**: Code treats single item as `Coherence = 1.0`. The penalty formula `(Coherence - 0.7) * 2.0` results in `(1.0 - 0.7) * 2 = 0.6`.
- **Fix Required**: Exceptions for N=1 in scoring logic.

### 3. Coherence Dilution (Ablation)
**Observation**: Removing "Ashwagandha" from the "Immunity Triad" *increased* the score (0.417 -> 0.551).
- **Interpretation**: Ashwagandha likely had low overlap with Amla/Guduchi, dragging down the *Average Pairwise Jaccard*.
- **Implication**: The current "Average Pairwise" metric penalizes broad, multi-target formulations (Coverage vs. Coherence tradeoff is unbalanced).

## Conclusion
The **Plausibility Scorer V1** is functional but naive.
- **Pros**: Successfully integrates graph queries (Evidence traceable).
- **Cons**: Metrics (Jaccard, Average) are brittle. Redundancy threshold is too high.
- **Next Step**: Switch from Average Jaccard to **Max Spanning Tree** or **Community Detection** tightness. Lower redundancy threshold or use vector cosine similarity (from Exp 001 embeddings) instead of strict Jaccard.
