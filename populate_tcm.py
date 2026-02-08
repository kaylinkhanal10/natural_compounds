import pandas as pd
import sqlite3
import os
import sys
from tqdm import tqdm

# Ensure backend path
sys.path.append(os.getcwd())
from backend.app.services.neo4j_service import Neo4jService

RAW_DIR = "raw_data"
CHEMBL_DB = "/mnt/natural_medicine_data/chembl_36/chembl_36_sqlite/chembl_36.db"

def main():
    print("--- TCM 2.0 Ingestion Pipeline ---")
    db = Neo4jService()
    
    # 0. Clean Slate (Optional but recommended for Full Ingestion)
    print("0. Cleaning old Herb/Compound nodes...")
    with db.driver.session() as session:
        session.run("MATCH (h:Herb) DETACH DELETE h")
        session.run("MATCH (c:Compound) DETACH DELETE c")

    # 1. Setup Constraints
    print("1. Setting up Schema Constraints...")
    with db.driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (h:Herb) REQUIRE h.herbId IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (h:Herb) REQUIRE h.scientificName IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Compound) REQUIRE c.compoundId IS UNIQUE")
        session.run("CREATE INDEX IF NOT EXISTS FOR (c:Compound) ON (c.name)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (c:Compound) ON (c.inchikey)")
    
    # 2. Ingest Herbs
    print("2. Ingesting Herbs (medicinal_material.xlsx)...")
    df_herbs = pd.read_excel(os.path.join(RAW_DIR, "medicinal_material.xlsx"))
    # Schema: LATIN, COMMON, KOREAN, ...
    
    # Generate mock IDs or use LATIN? 
    # Current MVP uses 'HERB_AMLA'. We will use 'HERB_{SANITIZED_LATIN}'.
    
    with db.driver.session() as session:
        # Pre-process and Deduplicate in Python
        unique_herbs = {}
        for _, row in df_herbs.iterrows():
             latin = str(row['LATIN']).strip() if pd.notna(row['LATIN']) else "Unknown"
             herb_id = f"HERB_{latin.upper().replace(' ', '_').replace('.', '')}"
             
             if herb_id not in unique_herbs:
                 unique_herbs[herb_id] = {
                     "herbId": herb_id,
                     "scientificName": latin,
                     "name": latin,
                     "pinyin": str(row.get('PINYIN', '')),
                     "chinese": str(row.get('CHINESE', ''))
                 }
        
        batch = []
        for hdata in unique_herbs.values():
             batch.append(hdata)
             
             if len(batch) >= 500:
                 session.run("""
                 UNWIND $batch as row
                 MERGE (h:Herb {scientificName: row.scientificName})
                 SET h.herbId = row.herbId,
                     h.name = row.name,
                     h.pinyin = row.pinyin,
                     h.chinese = row.chinese,
                     h.source = 'TCM 2.0'
                 """, batch=batch)
                 batch = []
        
        # Flush
        if batch:
             session.run("""
             UNWIND $batch as row
             MERGE (h:Herb {scientificName: row.scientificName})
             SET h.herbId = row.herbId,
                 h.name = row.name,
                 h.pinyin = row.pinyin,
                 h.chinese = row.chinese,
                 h.source = 'TCM 2.0'
             """, batch=batch)

    # 3. Ingest Compounds & Relations
    print("3. Ingesting Compounds (medicinal_compound.xlsx)...")
    df_comp = pd.read_excel(os.path.join(RAW_DIR, "medicinal_compound.xlsx"))
    # Schema: LATIN, ID, COMPOUND
    # Relation: (Herb {scientificName: LATIN}) -> (Compound {compoundId: ID})
    
    with db.driver.session() as session:
        batch = []
        for _, row in tqdm(df_comp.iterrows(), total=len(df_comp)):
             latin = str(row['LATIN']).strip()
             cid = str(row['ID'])
             cname = str(row['COMPOUND'])
             
             batch.append({
                 "latin": latin,
                 "compoundId": cid,
                 "name": cname
             })
             
             if len(batch) >= 1000:
                 session.run("""
                UNWIND $batch as row
                 MERGE (c:Compound {compoundId: row.compoundId})
                 SET c.name = row.name,
                     c.structure_status = 'symbolic_only',
                     c.embedding_status = 'unavailable'
                 
                 WITH c, row
                 MATCH (h:Herb) WHERE toLower(h.scientificName) = toLower(row.latin)
                 MERGE (h)-[:HAS_COMPOUND]->(c)
                 """, batch=batch)
                 batch = []
                 
        if batch:
             session.run("""
             UNWIND $batch as row
             MERGE (c:Compound {compoundId: row.compoundId})
             SET c.name = row.name,
                 c.structure_status = 'symbolic_only',
                 c.embedding_status = 'unavailable'
             
             WITH c, row
             MATCH (h:Herb) WHERE toLower(h.scientificName) = toLower(row.latin)
             MERGE (h)-[:HAS_COMPOUND]->(c)
             """, batch=batch)
             
    # 4. Hydrate Structures (Property File)
    print("4. Hydrating Structures (chemical_property.xlsx)...")
    df_prop = pd.read_excel(os.path.join(RAW_DIR, "chemical_property.xlsx"))
    
    with db.driver.session() as session:
        batch = []
        for _, row in df_prop.iterrows():
             if pd.notna(row['inchikey']) and pd.notna(row['molregno']):
                 batch.append({
                     "cid": str(row['molregno']), # Assuming this links to medicinal_compound.ID
                     "inchikey": str(row['inchikey'])
                 })
        
        if batch:
            session.run("""
            UNWIND $batch as row
            MATCH (c:Compound {compoundId: row.cid})
            SET c.inchikey = row.inchikey,
                c.structure_status = 'structure_backed'
            """, batch=batch)
            print(f"Updated {len(batch)} compounds with InChIKeys.")
            
    # 5. Resolve SMILES from ChEMBL
    print("5. Resolving SMILES from ChEMBL...")
    
    # Fetch InChIKeys where SMILES is missing
    with db.driver.session() as session:
        result = session.run("MATCH (c:Compound) WHERE c.inchikey IS NOT NULL AND c.smiles IS NULL RETURN c.inchikey as k")
        keys_to_resolve = [r['k'] for r in result]
        
    print(f"Resolving {len(keys_to_resolve)} keys...")
    
    if keys_to_resolve:
        conn = sqlite3.connect(CHEMBL_DB)
        cursor = conn.cursor()
        
        # Batch lookup
        updates = []
        chunk_size = 500
        for i in range(0, len(keys_to_resolve), chunk_size):
            chunk = keys_to_resolve[i:i+chunk_size]
            placeholders = ','.join(['?'] * len(chunk))
            
            q = f"SELECT standard_inchi_key, canonical_smiles FROM compound_structures WHERE standard_inchi_key IN ({placeholders})"
            cursor.execute(q, chunk)
            rows = cursor.fetchall()
            updates.extend(rows)
            
        conn.close()
        
        # Apply updates
        print(f"Found {len(updates)} SMILES.")
        with db.driver.session() as session:
             update_batch = [{"k": k, "s": s} for k, s in updates]
             # Batch set
             for i in range(0, len(update_batch), 1000):
                 b = update_batch[i:i+1000]
                 session.run("""
                 UNWIND $batch as row
                 MATCH (c:Compound {inchikey: row.k})
                 SET c.smiles = row.s,
                     c.structure_status = 'structure_backed'
                 """, batch=b)
                 
    print("Ingestion Complete.")
    db.close()

if __name__ == "__main__":
    main()
