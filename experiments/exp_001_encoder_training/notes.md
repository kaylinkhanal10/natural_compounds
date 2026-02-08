# Experiment 001: Supervised Molecular Encoder Training
**Date**: 2026-02-06
**Status**: Completed

## Objective
Train the `MoleculeEncoder` (GNN) on filtered ChEMBL data to map molecular graphs to biological targets.
Goal: Achieve reasonable `Recall@10` to prove the encoder learns meaningful biology, unlike the previous VAE which focused on chemical reconstruction.

## Configuration
- **Dataset**: ChEMBL v36 (SQLite)
- **Filters**: Single Protein targets, Confidence >= 7, Support >= 10 compounds.
- **Model**: GINEConv GNN (Hidden: 128)
- **Task**: Multi-label classification (BCEWithLogitsLoss)
- **Epochs**: 5
- **Limit**: 1000 compounds (Pilot run)

## Results
- **Training Loss**: decreased from 0.30 to 0.04.
- **Validation Recall@10**:
    - Start: ~9%
    - End: ~30.1%
    - Peak: ~33.0% (Epoch 3)

## Analysis
- **Learning Confirmed**: The model rapidly generalized to unseen validation compounds, reaching ~30% Recall@10 in just 5 epochs on a small subset.
- **Plausibility Enabler**: High recall suggests the embeddings cluster biologically similar compounds (target sharing), which is the prerequisite for the Plausibility Engine's "Coherence" logic.
- **Overfitting Warning**: Val Recall dipped slightly in Epoch 4-5 while Train Recall increased. Early stopping is necessary.

## Artifacts
- [Loss Curve](plots/loss_curve.png)
- [Retrieval Metrics](plots/retrieval_at_k.png)
