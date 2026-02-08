import os
import pickle
import numpy as np
from neo4j import GraphDatabase
import argparse
from dotenv import load_dotenv

# Load env for Neo4j credentials
load_dotenv('backend/.env')

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

class KGInitializer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def update_embeddings(self, embeddings_path):
        if not os.path.exists(embeddings_path):
            print(f"Embeddings file not found: {embeddings_path}")
            return

        print(f"Loading embeddings from {embeddings_path}...")
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            
        print(f"Updating {len(data)} compounds in Neo4j...")
        
        with self.driver.session() as session:
            # Update in batches
            batch_size = 1000
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                self._write_batch(session, batch)
                print(f"Processed {min(i+batch_size, len(data))}/{len(data)}")

    def _write_batch(self, session, batch):
        # Prepare valid JSON-serializable list
        clean_batch = []
        for item in batch:
            emb_list = item['embedding'].tolist() if isinstance(item['embedding'], np.ndarray) else item['embedding']
            clean_batch.append({
                'chembl_id': item['chembl_id'],
                'smiles': item['smiles'],
                'embedding': emb_list
            })
            
        query = """
        UNWIND $batch as row
        MERGE (c:Compound {chembl_id: row.chembl_id})
        ON CREATE SET c.smiles = row.smiles, c.embedding = row.embedding, c.source = 'ChEMBL36'
        ON MATCH SET c.embedding = row.embedding
        """
        session.run(query, batch=clean_batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str, default='raw_data/chembl_embeddings.pkl')
    args = parser.parse_args()
    
    kg = KGInitializer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        kg.update_embeddings(args.embeddings)
    finally:
        kg.close()
