import pandas as pd
import os

RAW_DIR = "/home/peridot/discovery/natural-medicine-ai/raw_data"
files = [
    "medicinal_material.xlsx",
    "medicinal_compound.xlsx",
    "prescription.xlsx",
    "chemical_protein.xlsx",
    "protein_disease.xlsx"
]

print("--- TCM 2.0 Schema Inspection ---\n")

for f in files:
    path = os.path.join(RAW_DIR, f)
    if os.path.exists(path):
        try:
            # Read just the header
            df = pd.read_excel(path, nrows=5)
            print(f"FILE: {f}")
            print(f"Columns: {list(df.columns)}")
            print(f"Sample Row: {df.iloc[0].to_dict()}")
            print("-" * 40)
        except Exception as e:
            print(f"FILE: {f} - Error reading: {e}")
    else:
        print(f"FILE: {f} - Not Found")
