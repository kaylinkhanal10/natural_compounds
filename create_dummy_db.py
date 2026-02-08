import sqlite3
import os
import numpy as np

def create_dummy_db(path):
    if os.path.exists(path):
        os.remove(path)
        
    conn = sqlite3.connect(path)
    c = conn.cursor()
    
    # Create Tables
    c.execute('''CREATE TABLE activities (
        molregno INTEGER,
        tid INTEGER,
        pchembl_value REAL,
        confidence_score INTEGER,
        standard_type TEXT,
        standard_value REAL,
        standard_units TEXT
    )''')
    
    c.execute('''CREATE TABLE compound_structures (
        molregno INTEGER,
        canonical_smiles TEXT,
        standard_inchi_key TEXT
    )''')
    
    c.execute('''CREATE TABLE target_dictionary (
        tid INTEGER,
        chembl_id TEXT,
        pref_name TEXT,
        target_type TEXT
    )''')
    
    # Insert Data
    # 50 Compounds, 5 Targets
    smiles_list = [
        "CCO", "CCN", "CCF", "CCCl", "CCBr",
        "c1ccccc1O", "c1ccccc1N", "c1ccccc1F", "c1ccccc1Cl", "c1ccccc1Br"
    ]
    
    targets = [f"CHEMBL{i}" for i in range(1, 6)]
    
    data_activities = []
    
    for i in range(100): # 100 activities
        molregno = i % 10 # 10 distinct compounds recycled
        smiles = smiles_list[molregno]
        inchi_key = f"INCHI_{molregno}"
        
        # Insert compound (ignore duplicates)
        c.execute("INSERT OR IGNORE INTO compound_structures VALUES (?, ?, ?)", (molregno, smiles, inchi_key))
        
        tid = (i % 5) + 1
        target_chembl = targets[tid-1]
        
        # Insert target
        c.execute("INSERT OR IGNORE INTO target_dictionary VALUES (?, ?, ?, ?)", (tid, target_chembl, f"Target {tid}", "SINGLE PROTEIN"))
        
        # Insert Activity
        pchembl = float(np.random.rand() * 10)
        c.execute("INSERT INTO activities VALUES (?, ?, ?, ?, ?, ?, ?)", 
                  (molregno, tid, pchembl, 9, "IC50", 10.0, "nM"))
                  
    conn.commit()
    conn.close()
    print(f"Created dummy DB at {path}")

if __name__ == "__main__":
    create_dummy_db("raw_data/chembl_dummy.db")
