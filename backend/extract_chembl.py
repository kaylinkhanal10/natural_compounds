import sqlite3
import pandas as pd
import os
import argparse
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def calculate_props(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        return {
            'mw': Descriptors.MolWt(mol),
            'exact_mw': Descriptors.ExactMolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'hba': Lipinski.NumHAcceptors(mol),
            'hbd': Lipinski.NumHDonors(mol),
            'rotb': Lipinski.NumRotatableBonds(mol),
            'atom_count': mol.GetNumAtoms(),
            'nring': Lipinski.RingCount(mol),
            'nrom': Lipinski.NumAromaticRings(mol),
        }
    except:
        return None

def extract_data(db_path, output_path, limit=None):
    if not os.path.exists(db_path):
        print(f"Error: DB not found at {db_path}")
        return

    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # OPTIMIZATION: Manual Join Strategy to avoid massive SQL Join
    
    # 1. Get Valid Assays (Confidence >= 7, Single Protein)
    print("Step 1: Filtering Assays...")
    # Get TIDs for SINGLE PROTEIN
    cursor.execute("SELECT tid FROM target_dictionary WHERE target_type = 'SINGLE PROTEIN'")
    valid_tids = {row[0] for row in cursor.fetchall()}
    
    # Get Assay IDs with Confidence >= 7 AND valid TID
    cursor.execute("SELECT assay_id, tid FROM assays WHERE confidence_score >= 7")
    valid_assays = [row[0] for row in cursor.fetchall() if row[1] in valid_tids]
    
    print(f"Found {len(valid_assays)} valid high-confidence assays.")
    
    if not valid_assays:
        print("No valid assays found.")
        conn.close()
        return

    # 2. Get Activities (Molregnos)
    print("Step 2: Fetching Active Compounds...")
    # Chunking assay IDs if too many
    # For now, just grab Activities where assay_id IN (...)
    # But IN (...) with 100k IDs is slow.
    
    # Better: Query activities and filter in Python? Activities is 15M rows. 
    # Or Temporary Table.
    
    cursor.execute("CREATE TEMP TABLE valid_assays_temp (assay_id INTEGER PRIMARY KEY)")
    cursor.executemany("INSERT INTO valid_assays_temp (assay_id) VALUES (?)", [(aid,) for aid in valid_assays])
    
    # Join Activities with Temp Table
    # DISTINCT molregno to minimize next step
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    query_act = f"""
    SELECT DISTINCT act.molregno 
    FROM activities act
    JOIN valid_assays_temp vat ON act.assay_id = vat.assay_id
    {limit_clause}
    """
    
    cursor.execute(query_act)
    valid_molregnos = [row[0] for row in cursor.fetchall()]
    print(f"Found {len(valid_molregnos)} unique active compounds.")
    
    # 3. Get Structures
    print("Step 3: Fetching Structures...")
    if not valid_molregnos:
        return
        
    cursor.execute("CREATE TEMP TABLE valid_mols_temp (molregno INTEGER PRIMARY KEY)")
    cursor.executemany("INSERT INTO valid_mols_temp (molregno) VALUES (?)", [(m,) for m in valid_molregnos])
    
    query_struct = """
    SELECT md.chembl_id, cs.canonical_smiles, cs.standard_inchi_key
    FROM compound_structures cs
    JOIN molecule_dictionary md ON cs.molregno = md.molregno
    JOIN valid_mols_temp vmt ON cs.molregno = vmt.molregno
    WHERE cs.canonical_smiles IS NOT NULL
    """
    
    df = pd.read_sql_query(query_struct, conn)
    conn.close()
    
    print(f"Extracted {len(df)} structures. Calculating properties...")
    
    # 4. RDKit Processing
    unique_mols = df.drop_duplicates(subset='standard_inchi_key')
    
    props_list = []
    # Print progress every 10%
    total = len(unique_mols)
    for idx, row in unique_mols.iterrows():
        p = calculate_props(row['canonical_smiles'])
        if p:
            p['chembl_id'] = row['chembl_id']
            p['inchikey'] = row['standard_inchi_key']
            p['smiles'] = row['canonical_smiles']
            props_list.append(p)
            
    final_df = pd.DataFrame(props_list)
    print(f"Valid Molecules with Properties: {len(final_df)}")
    
    print(f"Saving to {output_path}...")
    if output_path.endswith('.xlsx'):
        final_df.to_excel(output_path, index=False)
    else:
        final_df.to_csv(output_path, index=False)
        
    print("Extraction Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    
    extract_data(args.db, args.out, args.limit)
