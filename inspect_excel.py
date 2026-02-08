import pandas as pd
import os

def check_excel():
    df = pd.read_excel("raw_data/medicinal_compound.xlsx")
    print("Columns:", df.columns)
    
    # Filter for Glycyrrhiza
    mask = df['LATIN'].str.contains('Glycyrrhiza', case=False, na=False)
    matches = df[mask]['LATIN'].unique()
    
    print("\nUnique LATIN names containing 'Glycyrrhiza':")
    for m in matches:
        print(f"'{m}'")

    # Also check medicinal_material.xlsx
    print("\n--- medicinal_material.xlsx ---")
    df_mat = pd.read_excel("raw_data/medicinal_material.xlsx")
    mask_mat = df_mat['LATIN'].str.contains('Glycyrrhiza', case=False, na=False)
    matches_mat = df_mat[mask_mat]['LATIN'].unique()
    for m in matches_mat:
        print(f"'{m}'")

if __name__ == "__main__":
    check_excel()
