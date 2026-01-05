# Traditional Medicine World Model (TM-MC 2.0)

This module learns latent representations of natural compounds using a multi-modal VAE.



1. **Chemical Descriptors** (MW, LogP, TPSA, etc.)
2. **Protein Targets** (Multi-hot vector of biological targets)

## Purpose
This latent space represents the observed chemical–biological manifold of traditional-medicine compounds, not hypothetical drug space.

The goal is to provide a **continuous, 32-dimensional vector space** where proximity implies both chemical and biological similarity. These embeddings enable:
- **Similarity Search**: Find compounds chemically and biologically relatable.
- **Clustering**: Group compounds for exploration.
- **Diversity Sampling**: Select diverse compounds for screening.

> **SAFETY NOTICE**: This model is for **representation learning only**.
> - It does NOT predict efficacy or toxicity.
> - It does NOT predict binding affinity (IC50/Ki).
> - It does NOT generate new molecules.

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
python3 -m backend.app.ml.world_model.utils
```

## Architecture

- **Encoders**:
  - `ChemEncoder`: MLP transforming numeric descriptors.
  - `ProtEncoder`: MLP transforming sparse target vectors.
- **Fusion**: Concatenation -> MLP -> Latent Distribution (Gaussian).
- **Decoders**:
  - `ChemDecoder`: Reconstructs numeric descriptors (MSE Loss).
  - `ProtDecoder`: Reconstructs target vector (Weighted BCE Loss).
- **Loss**: $\mathcal{L} = \lambda_{chem}\mathcal{L}_{MSE} + \lambda_{prot}\mathcal{L}_{BCE} + \beta D_{KL}$

## Outputs
Embeddings are used only for relative comparison, clustering, and diversity analysis.

- **Embeddings**: stored in Neo4j on `Compound` nodes.
- **Metrics**: Saved to `checkpoints/metrics.json`.
