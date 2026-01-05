import os
import argparse
from neo4j import GraphDatabase
from .infer import WorldModelInference
import yaml

class Neo4jIntegrator:
    def __init__(self, config_path='backend/app/ml/world_model/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        uri = self.config['graph']['uri']
        user = self.config['graph']['user']
        password = self.config['graph']['password']
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def update_embeddings(self, inference_service):
        print("Starting Neo4j update...")
        # Get all embeddings
        if inference_service.embeddings is None:
            inference_service._compute_all_embeddings()
            
        embeddings = inference_service.embeddings
        compound_ids = inference_service.compound_ids
        
        # Batch update
        batch_size = 1000
        total = len(compound_ids)
        
        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch_ids = compound_ids[i:i+batch_size]
                batch_embs = embeddings[i:i+batch_size].tolist()
                
                # Prepare data format: [{id: '...', embedding: [...]}, ...]
                data = [{'inchikey': cid, 'embedding': emb} for cid, emb in zip(batch_ids, batch_embs)]
                
                query = """
                UNWIND $batch as row
                MATCH (c:Compound {inchikey: row.inchikey})
                SET c.embedding = row.embedding
                """
                session.run(query, batch=data)
                print(f"Updated {min(i+batch_size, total)}/{total} compounds")
                
        print("Neo4j update complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='backend/app/ml/world_model/config.yaml')
    args = parser.parse_args()
    
    # 1. Load Inference
    inference = WorldModelInference(config_path=args.config)
    
    # 2. Connect to Graph
    integrator = Neo4jIntegrator(config_path=args.config)
    
    # 3. Push
    try:
        integrator.update_embeddings(inference)
    finally:
        integrator.close()

if __name__ == '__main__':
    main()
