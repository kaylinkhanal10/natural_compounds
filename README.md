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
