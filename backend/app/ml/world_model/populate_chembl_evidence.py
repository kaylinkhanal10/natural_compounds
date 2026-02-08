import sqlite3
import os
import sys
from tqdm import tqdm
from backend.app.services.neo4j_service import Neo4jService

# Add project root to path
sys.path.append(os.getcwd())

def populate_chembl_data(sqlite_path, limit=None):
    if not os.path.exists(sqlite_path):
        print(f"Error: DB not found at {sqlite_path}")
        return

    print("Connecting to Neo4j...")
    neo = Neo4jService()
    
    # 1. Get all Compound InChIKeys from Neo4j
    print("Fetching compounds from Neo4j...")
    with neo.driver.session() as session:
        result = session.run("MATCH (c:Compound) WHERE c.inchikey IS NOT NULL RETURN c.inchikey as k, elementId(c) as id")
        # Map InChIKey -> NodeID (or just use InChIKey for matching)
        # We'll use InChIKey to match in standard Cypher queries
        compounds = {record['k']: record['id'] for record in result}
        
    print(f"Found {len(compounds)} compounds in Neo4j.")
    
    if len(compounds) == 0:
        return

    # 2. Query ChEMBL for these keys
    print("Querying ChEMBL for activities...")
    conn = sqlite3.connect(sqlite_path)
    
    # We can't query 22k keys in IN clause easily.
    # Better to attach the ChEMBL DB? Or chunk it?
    # Or just iterate matches if we can index?
    # ChEMBL `compound_structures` has `standard_inchi_key`.
    # Index likely exists on `standard_inchi_key`.
    
    keys = list(compounds.keys())
    chunk_size = 1000
    
    total_edges = 0
    
    for i in range(0, len(keys), chunk_size):
        chunk = keys[i:i+chunk_size]
        placeholders = ','.join(['?'] * len(chunk))
        
        query = f"""
        SELECT 
            cs.standard_inchi_key,
            td.chembl_id,
            td.pref_name,
            act.pchembl_value,
            act.standard_type,
            d.pubmed_id,
            d.doi
        FROM compound_structures cs
        JOIN activities act ON cs.molregno = act.molregno
        JOIN assays ass ON act.assay_id = ass.assay_id
        JOIN target_dictionary td ON ass.tid = td.tid
        LEFT JOIN docs d ON act.doc_id = d.doc_id
        WHERE cs.standard_inchi_key IN ({placeholders})
        AND td.target_type = 'SINGLE PROTEIN'
        AND act.pchembl_value IS NOT NULL
        """
        
        cursor = conn.execute(query, chunk)
        rows = cursor.fetchall()
        
        if not rows:
            continue
            
        # Group by Target to minimize write transactions
        # Or just write rows.
        
        with neo.driver.session() as session:
            # Batch write
            # Create Targets and Edges
            
            # Prepare list of dicts for UNWIND
            batch_data = []
            for row in rows:
                ikey, chembl_id, target_name, pval, type_, pmid, doi = row
                
                batch_data.append({
                    'inchikey': ikey,
                    'target_id': chembl_id,
                    'target_name': target_name or chembl_id,
                    'pchembl': float(pval),
                    'type': type_,
                    'pubmed': str(pmid) if pmid else None,
                    'doi': str(doi) if doi else None
                })
                
            q_write = """
            UNWIND $batch as row
            
            MATCH (c:Compound {inchikey: row.inchikey})
            
            MERGE (t:Protein {chembl_id: row.target_id})
            ON CREATE SET t.name = row.target_name
            
            MERGE (c)-[r:TARGETS]->(t)
            ON CREATE SET r.pchembl = row.pchembl, r.type = row.type, r.source = 'ChEMBL'
            ON MATCH SET r.pchembl = row.pchembl // Update if needed
            
            WITH c, t, row
            WHERE row.pubmed IS NOT NULL
            
            MERGE (e:Evidence {pubmed_id: row.pubmed})
            ON CREATE SET e.source = 'PubMed', e.url = 'https://pubmed.ncbi.nlm.nih.gov/' + row.pubmed
            
            MERGE (c)-[:SUPPORTED_BY]->(e)
            MERGE (t)-[:SUPPORTED_BY]->(e)
            """
            
            session.run(q_write, batch=batch_data)
            total_edges += len(batch_data)
            
        if limit and total_edges >= limit:
            break
            
        print(f"Processed {i + len(chunk)}/{len(keys)} compounds. Added {len(rows)} edges.")
        
    conn.close()
    neo.close()
    print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default='/mnt/natural_medicine_data/chembl_36/chembl_36_sqlite/chembl_36.db')
    args = parser.parse_args()
    
    # Adjust path if extraction structure is different
    # Just generic path for now
    populate_chembl_data(args.db_path)
