import os
import sys

# Ensure backend module can be found
sys.path.append(os.getcwd())

from backend.app.ml.world_model.infer import WorldModelInference
from backend.app.services.neo4j_service import Neo4jService

def main():
    print("Initializing World Model Inference...")
    try:
        # Config path assuming running from root
        infer = WorldModelInference(config_path='backend/app/ml/world_model/config.yaml')
    except Exception as e:
        print(f"Failed to load VAE: {e}")
        return

    print("Connecting to Neo4j...")
    neo4j = Neo4jService()
    
    print("Computing and Storing Embeddings...")
    count = 0
    with neo4j.driver.session() as session:
        # We process in batches if needed, but simple loop is fine for MVP
        for i, inchikey in enumerate(infer.compound_ids):
            embedding = infer.embeddings[i].tolist() # Convert numpy to list
            
            # Update query
            q = """
            MATCH (c:Compound {inchikey: $inchikey})
            SET c.embedding = $embedding
            RETURN count(c) as updated
            """
            res = session.run(q, inchikey=inchikey, embedding=embedding).single()
            if res and res['updated'] > 0:
                count += 1
            
            if (i+1) % 100 == 0:
                print(f"Processed {i+1}/{len(infer.compound_ids)} compounds...")
                
    print(f"Finished. Updated {count} compounds with embeddings in Neo4j.")
    neo4j.close()

if __name__ == "__main__":
    main()
