# SynerG: Synergistic Graph Intelligence (MVP)

> **Scientific Validation for Natural Medicine.**

SynerG is a B2B demo platform that bridges the gap between traditional herbal wisdom and modern pharmacological science. It uses a **Neuro-Symbolic** approach to reason about formulations, interactions, and synergy.

---

## 🧠 The Architecture
SynerG combines two forms of AI to solve the "Black Box" problem:

1.  **The Neural Brain (World Model)**: A **Variational Autoencoder (VAE)** that understands the continuous landscape of chemistry. It builds a "map" where distance = difference in properties.
    *   *See [World Model Documentation](backend/app/ml/world_model/README.md) for architecture, plots, and methodology.*
2.  **The Symbolic Brain (Knowledge Graph)**: A **Neo4j** database that stores discrete biological facts (Herbs $\to$ Compounds $\to$ Proteins $\to$ Effect).

**Synergy Score**: We combine these two to mathematically score herb combinations:
$$ \text{Synergy} = \text{Chemical Diversity (AI)} + \text{Target Focus (Graph)} $$

---

## 🚀 Setup Instructions

### 1. Database (Neo4j)
Start the Graph Database:
```bash
cd docker
docker-compose up -d
```
Credentials: `neo4j` / `password`.

### 2. Backend (FastAPI)
The brain of the operation.
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API Docs: `http://localhost:8000/docs`

### 3. Frontend (Next.js)
The interface for discovery.
```bash
cd frontend
npm install
npm run dev
```
App: `http://localhost:3000`

---

## ⚡ Key Features
- **Herb Explorer**: Detailed profiles with verified chemical data.
- **Combination Reasoning**: AI-driven analysis of how herbs interact (Synergy Scores, Consensus Maps).
- **Formula Explorer**: Browse traditional formulations with mechanistic explanations.
- **Disease Network**: Explore the "Common Enemy" shared targets.

## 📊 Validation
The system includes verifyable ingestion pipelines and transparent AI metrics.
Check `backend/app/ml/world_model/README.md` for the latest training loss curves and reconstruction accuracy ($R^2 > 0.95$).

---

## 🔬 Scientific Core Principles

### 1. Lipinski's Rule of 5 (Bioavailability)
We filter compounds based on this famous pharmaceutical rule to ensure we recommend **absorbable** medicine, not just "fiber."

> **"If you can't absorb it, it can't heal you."**

*   **MW < 500**: Heavy molecules (like Tannins, MW > 800) often stay in the gut. Small ones (Curcumin, MW 368) can enter the blood.
*   **LogP < 5**: Too oily? Stuck in fat. Too watery? Won't cross membranes. We look for the "Goldilocks" zone.
*   **H-Bonds**: Too many sticky atoms prevent passage through cell walls.

**Why it matters**: Herbs like **Tumeric (Curcumin)** pass the MW check but struggle with metabolism. This is why our system checks for "Helper Herbs" (like Black Pepper) to boost bioavailability.

### 2. "Lock and Key" Mechanism (Shared Targets)
When you see a "Consensus Map" or "Shared Targets" in SynerG, we are calculating the **Biological Overlap**.

*   **The Lock**: The Protein (e.g., **TNF-Alpha**, the inflammation switch).
*   **The Keys**: The Compounds inside the herbs.
*   **The Hit**: When a compound fits the protein.

**Interpreting the Score**:
*   **Many Shared Targets (> 50)**: **Reinforcement**. The herbs are "singing the same song," hitting the same switches. This is powerful but can be redundant.
*   **Few Shared Targets**: **Complementarity**. One herb stops the fire (Inflammation), the other repairs the wall (Regeneration).
*   **The Sweet Spot**: We aim for high hits on key disease outcomes (like Inflammation determinants) while maintaining chemical diversity.
