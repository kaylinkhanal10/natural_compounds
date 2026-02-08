# OsadAI Experiment Index

**Master Record of Experiments**

| ID | Status | Name | Purpose | Key Findings |
| :--- | :--- | :--- | :--- | :--- |
| **EXP-001** | **Completed** | Supervised Encoder Training | Verify GNN learns biological targets | High Recall@10 (~30%), proves strong biological signal. |
| **EXP-002** | **Completed** | Plausibility Logic Verification (Herb Level) | Sanity check scoring logic | **FAIL**: Redundancy penalty ineffective; Single herbs penalized. |
| **EXP-003** | **In Progress** | Compound-Level Plausibility V2 | Validate Compound Reasoning | |

---

## EXP-001: Supervised Molecular Encoder (ChEMBL)
**Objective**: Train GNN on filtered ChEMBL dataset.
*   **Status**: Success.

## EXP-002: Plausibility Scorer Sanity Checks
**Objective**: Verify heuristic logic (Herb-Level).
*   **Status**: Mixed/Fail (Failures pivoted to learning for EXP-003).

## EXP-003: Compound-Level Plausibility Logic
**Objective**: Validate refined Plausibility Engine operating at Compound Level.
*   **Key Shifts**:
    *   **Unit of Reasoning**: Herb -> Set of Compounds.
    *   **Expansion**: Top-20 Compounds by Target Degree.
    *   **Redundancy**: `Bio_Overlap * Chem_Similarity`.
    *   **Aggregation**: `Max(Pairwise_Set_Scores)` to avoid dilution.
*   **Test Script**: [test_compound_logic.py](exp_003_compound_logic/test_compound_logic.py)
*   **Results**: [scores.csv](exp_003_compound_logic/scores.csv)
*   **Status**: **Mixed Success**.
    *   **Architecture Verified**: Scoring now correctly processes compound sets and uses embeddings.
    *   **Baseline Valid**: Single Herb = 1.0 (No self-redundancy).
    *   **Metric limitation**: *Turmeric + Pippali* (0.379) scored lower than *Turmeric* (1.0).
        *   *Reasoning*: Pippali acts via **bioavailability enhancement** (PK), not target overlap (PD). The current "Target Jaccard" metric correctly sees little overlap, thus low coherence.
*   **Limits**: Target Overlap logic misses Pharmacokinetic (Bioavailability) synergy (Turmeric + Pippali).

## EXP-004: Pure Compound Interactions
**Objective**: Validate Scorer with direct Molecular Inputs (No Herbs).
*   **Test Script**: [test_pure_compounds.py](exp_004_pure_compounds/test_pure_compounds.py)
*   **Results**: [scores.csv](exp_004_pure_compounds/scores.csv)
*   **Status**: **Architecture Success**.
    *   **Pure Input**: Scorer now accepts `score_direct(compound_sets)`.
    *   **Curcumin + Piperine** (Score 0.03): Correctly detects lack of target overlap (Bio Sim = 0). Confirms need for PK rules.
    *   **Curcumin + Quercetin** (Score 0.00): Penalized for High Redundancy (High Bio Overlap + High Chem Sim).
        *   *Insight*: Strong target overlap + structural similarity = Redundancy in this model. To see them as "Synergistic", we must distinguish their *downstream* effects (Pathway Coherence vs Target Identity).

## EXP-005: Pharmacokinetic (PK) Rules
**Objective**: Solve the "Pippali Paradox" (Synergy via bioavailability, not just targets).
*   **Method**: Implemented `_is_pk_enhancer` check for CYP3A4/P-gp inhibitors. Applied +0.3 Coherence Boost.
*   **Test Script**: [test_pk_logic.py](exp_005_pk_rules/test_pk_logic.py)
*   **Results**:
    *   **Curcumin + Piperine**: Score increased from **0.03** $\rightarrow$ **0.16**.
    *   **Mechanism**: System detected Piperine targets `CYP3A4/ABCB1` and applied synergy boost.
    *   **Status**: **Success**. The engine now accounts for "Helper Herbs".

