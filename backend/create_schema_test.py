import sqlite3
import os

DB_PATH = "chembl_schema_test.db"

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 1. molecule_dictionary
cursor.execute('''
    CREATE TABLE molecule_dictionary (
        molregno INTEGER PRIMARY KEY,
        chembl_id TEXT,
        pref_name TEXT,
        structure_type TEXT,
        max_phase INTEGER
    )
''')

# 2. compound_structures
cursor.execute('''
    CREATE TABLE compound_structures (
        molregno INTEGER PRIMARY KEY,
        canonical_smiles TEXT,
        standard_inchi_key TEXT,
        FOREIGN KEY(molregno) REFERENCES molecule_dictionary(molregno)
    )
''')

# 3. activities
cursor.execute('''
    CREATE TABLE activities (
        activity_id INTEGER PRIMARY KEY,
        assay_id INTEGER,
        molregno INTEGER,
        standard_type TEXT,
        standard_value REAL,
        standard_units TEXT,
        standard_relation TEXT,
        pchembl_value REAL,
        FOREIGN KEY(molregno) REFERENCES molecule_dictionary(molregno)
        FOREIGN KEY(assay_id) REFERENCES assays(assay_id)
    )
''')

# 4. assays
cursor.execute('''
    CREATE TABLE assays (
        assay_id INTEGER PRIMARY KEY,
        tid INTEGER,
        description TEXT,
        assay_type TEXT,
        confidence_score INTEGER,
        FOREIGN KEY(tid) REFERENCES target_dictionary(tid)
    )
''')

# 5. target_dictionary
cursor.execute('''
    CREATE TABLE target_dictionary (
        tid INTEGER PRIMARY KEY,
        chembl_id TEXT,
        pref_name TEXT,
        target_type TEXT
    )
''')

# Insert Dummy Data (Target + Assay)
cursor.execute("INSERT INTO target_dictionary (tid, pref_name, target_type) VALUES (1, 'Target A', 'SINGLE PROTEIN')")
cursor.execute("INSERT INTO assays (assay_id, tid, assay_type, confidence_score) VALUES (10, 1, 'B', 9)")

# Insert 20 Dummy Molecules for VAE Testing
smiles_list = [
    "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "COc1cc(C=O)ccc1O",
    "C1=CC=C(C=C1)C=O", "CC(=O)NO", "CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO",
    "CN", "CCN", "CCCN", "CCCCN", "CCCCCN", "C(=O)O", "CC(=O)O", "CCC(=O)O", "CCCC(=O)O", "CCCCC(=O)O"
]

for i, smi in enumerate(smiles_list):
    mr = 100 + i
    cid = f"CHEMBL{mr}"
    ikey = f"INCHI_{mr}"
    
    cursor.execute("INSERT INTO molecule_dictionary (molregno, chembl_id) VALUES (?, ?)", (mr, cid))
    cursor.execute("INSERT INTO compound_structures (molregno, canonical_smiles, standard_inchi_key) VALUES (?, ?, ?)", (mr, smi, ikey))
    # Link to Assay 10 (Confidence 9) -> Required for filter
    cursor.execute("INSERT INTO activities (activity_id, assay_id, molregno, standard_type, standard_value, standard_units) VALUES (?, 10, ?, 'IC50', 10.0, 'nM')", (1000+i, mr))

conn.commit()
conn.close()
print(f"Schema test DB created: {DB_PATH} with {len(smiles_list)} molecules")
