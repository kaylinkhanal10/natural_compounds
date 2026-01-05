
import pandas as pd
import os

data_dir = "../raw_data"
files = [
    "chemical_property.xlsx",
    "chemical_protein.xlsx",
    "protein_disease.xlsx",
    "medicinal_compound.xlsx",
    "medicinal_material.xlsx",
    "prescription.xlsx"
]

for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        try:
            # Read just the header
            df = pd.read_excel(path, nrows=0)
            print(f"--- {f} ---")
            print(list(df.columns))
            print("\n")
        except Exception as e:
            print(f"Error reading {f}: {e}")
    else:
        print(f"File not found: {f}")
