import pandas as pd
import sqlite3
import os

RAW_DIR = "raw_data"
DB_PATH = "backend/data/chembl/chembl_36.db" # Check actual path
# User said: /mnt/natural_medicine_data/chembl_36/chembl_36_sqlite/chembl_36.db
REAL_DB_PATH = "/mnt/natural_medicine_data/chembl_36/chembl_36_sqlite/chembl_36.db"

def check_linkage():
    print("Loading Dataframes...")
    df_comp = pd.read_excel(os.path.join(RAW_DIR, "medicinal_compound.xlsx"))
    df_prop = pd.read_excel(os.path.join(RAW_DIR, "chemical_property.xlsx"))
    
    print(f"Compounds: {len(df_comp)} rows. IDs: {df_comp['ID'].min()} - {df_comp['ID'].max()}")
    print(f"Properties: {len(df_prop)} rows. Molregnos: {df_prop['molregno'].min()} - {df_prop['molregno'].max()}")
    
    # Check Intersection
    comp_ids = set(df_comp['ID'])
    prop_ids = set(df_prop['molregno'])
    
    overlap = comp_ids.intersection(prop_ids)
    print(f"Overlap Count: {len(overlap)}")
    
    if len(overlap) == 0:
        print("CRITICAL: No overlap between Compound ID and Property Molregno!")
        # Check first few
        print("Comp IDs head:", list(df_comp['ID'].head()))
        print("Prop Regs head:", list(df_prop['molregno'].head()))
    else:
        print("Linkage Verified.")

    # Check InChIKey overlap with ChEMBL
    print("\nChecking ChEMBL Resolution...")
    inchikeys = df_prop['inchikey'].dropna().unique()
    print(f"Unique InChIKeys in TCM: {len(inchikeys)}")
    
    if os.path.exists(REAL_DB_PATH):
        conn = sqlite3.connect(REAL_DB_PATH)
        cursor = conn.cursor()
        
        # Check a sample
        sample_keys = inchikeys[:100]
        placeholders = ','.join(['?'] * len(sample_keys))
        query = f"""
            SELECT count(canonical_smiles) 
            FROM compound_structures 
            WHERE standard_inchi_key IN ({placeholders})
        """
        cursor.execute(query, list(sample_keys))
        count = cursor.fetchone()[0]
        print(f"Found {count} SMILES for 100 sample TCM InChIKeys in ChEMBL.")
        conn.close()
    else:
        print(f"DB not found at {REAL_DB_PATH}")

if __name__ == "__main__":
    check_linkage()
