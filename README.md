# AI-Native Natural Medicine Discovery Platform (MVP)

A demo-grade B2B platform for evidence-backed reasoning over natural medicine formulations.

## Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Node.js 18+

## Setup Instructions

### 1. Database (Neo4j)
Start the Neo4j container:
```bash
cd docker
docker-compose up -d
```
Credentials: `neo4j` / `password`.

**Data Ingestion (Phase 2)**:
The platform now includes a robust ingestion pipeline for Excel datasets.
1. Place your `.xlsx` files in `raw_data/` at the project root.
2. Run the ingestion pipeline (converts to CSV, dedupes, and imports to Neo4j):
   ```bash
   # From natural-medicine-ai root
   cd backend
   python3 -m app.ingest.run_ingestion
   cd ..
   docker cp backend/app/data/staging/. natural-medicine-neo4j:/var/lib/neo4j/import/
   docker exec -i natural-medicine-neo4j cypher-shell -u neo4j -p password < backend/app/scripts/reset.cypher
   docker exec -i natural-medicine-neo4j cypher-shell -u neo4j -p password < backend/app/scripts/import_neo4j.cypher
   docker exec -i natural-medicine-neo4j cypher-shell -u neo4j -p password < backend/app/scripts/load_from_csv.cypher
   ```

### 2. Backend
Navigate to `backend/`:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API: `http://localhost:8000/docs`.

### 3. Frontend
Navigate to `frontend/`:
```bash
cd frontend
npm install
npm run dev
```
App: `http://localhost:3000`.

## Features
- **Herb Explorer**: Detailed profiles with Compounds (MW, LogP), Targets (Proteins), and Diseases.
- **Combination Reasoning**: Synergy analysis showing shared protein targets and gene pathways.
- **Formula Explorer**: Browse traditional formulations, ingredients, and mechanistic rationale.
- **Disease Network**: Explore diseases and their associated proteins and potential herbal treatments.
- **Intent Search**: Natural language query mapping.

## Validation
- **Ingestion Logs**: See `backend/app/data/logs/ingestion_summary.json`.
- **Sample Query**: `GET /formulas/` to see ingested prescriptions.
