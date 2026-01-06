# SynerG (Synergistic Graph Intelligence)
## World Model for Natural Medicine Discovery

**SynerG** allows us to scientifically decode the "Entourage Effect" of ancient herbal formulas.
This module learns latent representations of natural compounds using a custom Variational Autoencoder (VAE), creating a "Chemical Map" for calculating **Synergy Scores**.



1. **Chemical Descriptors** (MW, LogP, TPSA, etc.)

## Purpose
This latent space represents the observed chemical manifold of natural-medicine compounds.

The goal is to provide a **continuous, 32-dimensional vector space** where proximity implies chemical similarity. 

> **SAFETY NOTICE**: This model is for **representation learning only**.
> - It does NOT predict efficacy or toxicity.
> - It does NOT predict binding affinity (IC50/Ki).
> - It does NOT generate new molecules.
> - **No biological targets are predicted by the neural model.** All biological claims are grounded in curated knowledge graph paths.

## Usage

### 1. Training
Train the VAE on the provided Excel datasets:
```bash
python3 -m backend.app.ml.world_model.train
```
Configuration is managed in `config.yaml`.

### 2. Inference
Load the trained model and generate embeddings:
```python
from backend.app.ml.world_model.infer import WorldModelInference

service = WorldModelInference()
# Get embedding for a specific InChIKey
vector = service.get_embedding('SOME_INCHIKEY_STRING')
# Find nearest neighbors
neighbors = service.find_nearest_neighbors('SOME_INCHIKEY_STRING', k=5)
```

### 3. Graph Integration
Write learned embeddings back to Neo4j (`Compound.embedding` property):
```bash
python3 backend/app/ml/world_model/populate_neo4j_embeddings.py
```

## Architecture

- **Encoder**:
  - `ChemEncoder`: MLP transforming numeric descriptors.
- **Latent Space**: 32-dimensional Gaussian.
- **Decoder**:
  - `ChemDecoder`: Reconstructs numeric descriptors (MSE Loss).
- **Loss**: $\mathcal{L} = \mathcal{L}_{MSE} + \beta D_{KL}$

## Outputs
Embeddings are used only for relative comparison, clustering, and diversity analysis.

- **Embeddings**: stored in Neo4j on `Compound` nodes.
- **Metrics**: Saved to `checkpoints/metrics.json`.
