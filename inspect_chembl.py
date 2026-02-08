import sqlite3
import pandas as pd

DB_PATH = '/mnt/natural_medicine_data/chembl_36/chembl_36_sqlite/chembl_36.db'

def inspect():
    conn = sqlite3.connect(DB_PATH)
    try:
        # Check tables
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        print("Tables found:", tables['name'].tolist())
        
        # Check activities columns
        print("\n--- activities columns ---")
        try:
            cols = pd.read_sql("PRAGMA table_info(activities);", conn)
            # Print ALL columns
            for _, row in cols.iterrows():
                print(f"{row['cid']}: {row['name']} ({row['type']})")
        except Exception as e:
            print(f"Error reading activities: {e}")

        # Check assays columns
        print("\n--- assays columns ---")
        try:
            cols = pd.read_sql("PRAGMA table_info(assays);", conn)
            for _, row in cols.iterrows():
                print(f"{row['cid']}: {row['name']} ({row['type']})")
        except Exception as e:
            print(f"Error reading assays: {e}")

        # Check target_dictionary columns
        print("\n--- target_dictionary columns ---")
        try:
            cols = pd.read_sql("PRAGMA table_info(target_dictionary);", conn)
            print(cols[['name', 'type']])
        except Exception as e:
            print(f"Error reading target_dictionary: {e}")

    finally:
        conn.close()

if __name__ == "__main__":
    inspect()
