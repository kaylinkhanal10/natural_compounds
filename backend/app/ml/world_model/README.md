# OsadAI World Model: The Neuro-Symbolic Core

**OsadAI (Synergistic Graph Intelligence)** uses a hybrid AI approach to decode natural medicine. The **World Model** is the "Neural" half of this brain, responsible for understanding the continuous landscape of chemical structures.

## 1. Methodology
We treat chemistry not as discrete labels, but as a continuous mathematical manifold.

### Why Variational Autoencoder (VAE)?
Unlike standard Autoencoders that "memorize" inputs, a VAE learns a **probability distribution**.
*   **Autoencoder**: Good for compression, but the latent space is "gappy." Interpolating between two molecules generates nonsense.
*   **VAE**: Enforces a smooth Gaussian structure. This ensures that **distance in latent space = difference in chemical properties.**

### Mechanism
1.  **Input**: 2048-bit Extended Connectivity Fingerprint (ECFP) representing the molecule's structure.
2.  **Encoder**: Compresses this high-dimensional vector into a 32-dimensional Gaussian distribution ($\mu, \sigma$).
3.  **Reparameterization Trick**: We sample $z = \mu + \sigma \cdot \epsilon$ to allow backpropagation through random nodes.
4.  **Decoder**: Reconstructs the original fingerprint from $z$.

## 2. Architecture & Hyperparameters
The model is a lightweight, strictly-regularized MLP designed for small-data stability.

```python
# Architecture Overview
Input (2048) 
  -> Linear(256) -> LayerNorm -> GELU -> Dropout(0.1)
  -> Linear(128) -> LayerNorm -> GELU 
  -> Latent(32) (Mu + LogVar)
  -> Reparameterization (z)
  -> Linear(128) -> LayerNorm -> GELU
  -> Linear(256) -> LayerNorm -> GELU
  -> Output (2048) -> MSE Loss
```

*   **Latent Dimension**: 32 (Compression Ratio ~64:1)
*   **Activation**: GELU (Smoother gradients than ReLU)
*   **Normalization**: LayerNorm (Essential for chemical feature stability)

## 3. Synergy Score Calculation
The core innovation of OsadAI is the formula quantifying how well two herbs work together.

$$ \text{Score} = (\alpha \cdot D_{chem}) + (\beta \cdot C_{targets}) - (\gamma \cdot R_{redundancy}) $$

*   **$D_{chem}$ (Chemical Distance)**: Euclidean distance in the VAE latent space. We reward *difference* (Diversity).
*   **$C_{targets}$ (Target Consensus)**: Number of shared biological targets (symbolic graph overlap). We reward *focus*.
*   **$R_{redundancy}$ (Redundancy)**: Penalty for being too chemically similar without adding new targets.

## 4. Evaluation Metrics
We track three key metrics to ensure the model is learning a valid "World Map" of chemistry.

### A. Total Loss (Convergence)
The total cost paid by the model. The rise in loss over time is due to **KL Annealing** (gradually turning on the tax for using the latent space), forcing the model to organize its memory.
![Loss Curve](checkpoints/figures/loss_curve.png)

### B. Reconstruction Quality ($R^2$)
Measures how accurately the model can "remember" a molecule.
*   **Goal**: $> 0.90$
*   **Current status**: Achieving ~0.96. The model effectively captures chemical identity.
![R2 Score](checkpoints/figures/chem_r2.png)

### C. Latent Organization (KL Divergence)
Measures how "smooth" the chemical map is.
*   **Goal**: Non-zero but stable.
*   **Significance**: A healthy KLD proves the map is navigable and not just memorized (Posterior Collapse).
![KL Divergence](checkpoints/figures/kld.png)

## 5. Usage
To generate embeddings for the Graph:
```bash
python3 backend/app/ml/world_model/populate_neo4j_embeddings.py
```
This script acts as the bridge, injecting the Neural intuition (Embeddings) into the Symbolic Brain (Neo4j).
