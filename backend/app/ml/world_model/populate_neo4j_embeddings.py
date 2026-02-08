import os
import sys
from tqdm import tqdm

# Ensure backend module can be found
sys.path.append(os.getcwd())

from backend.app.ml.world_model.infer_supervised import SupervisedInference
from backend.app.services.neo4j_service import Neo4jService

def main():
    print("Initializing Supervised Inference...")
    try:
        infer = SupervisedInference(config_path='backend/app/ml/world_model/config.yaml')
    except Exception as e:
        print(f"Failed to load Model: {e}")
        return

    print("Connecting to Neo4j...")
    neo4j = Neo4jService()
    
    print("Fetching Compounds without new embeddings...")
    # Optionally force update all
    q_fetch = "MATCH (c:Compound) WHERE c.smiles IS NOT NULL RETURN c.inchikey as k, c.smiles as s"
    
    compounds = []
    with neo4j.driver.session() as session:
        result = session.run(q_fetch)
        compounds = [record.data() for record in result]
        
    print(f"Found {len(compounds)} compounds to process.")
    
    batch_size = 100
    
    with neo4j.driver.session() as session:
        for i in tqdm(range(0, len(compounds), batch_size)):
            batch = compounds[i:i+batch_size]
            smiles = [b['s'] for b in batch]
            keys = [b['k'] for b in batch]
            
            embeddings = infer.encode(smiles)
            
            update_list = []
            for k, emb in zip(keys, embeddings):
                if emb is not None:
                    update_list.append({'k': k, 'e': emb.tolist()})
            
            if update_list:
                q_update = """
                UNWIND $updates as row
                MATCH (c:Compound {inchikey: row.k})
                SET c.embedding_supervised = row.e
                """
                session.run(q_update, updates=update_list)
                
    print("Done.")
    neo4j.close()

if __name__ == "__main__":
    main()
