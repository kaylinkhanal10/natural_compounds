import pandas as pd
import os
import logging
import json
import yaml
from .compound_resolver import CompoundResolver
from .normalize import clean_float, clean_int

logger = logging.getLogger(__name__)

class ChemicalPropertyIngest:
    def __init__(self, file_path, config, output_dir):
        self.file_path = file_path
        self.config = config
        self.output_dir = output_dir
        self.resolver = CompoundResolver()
        self.stats = {
            "total_rows_read": 0,
            "unique_compounds": 0,
            "skipped_rows": 0,
            "resolution_methods": {"INCHIKEY": 0, "CID": 0, "SMILES": 0, "SOURCE_ID": 0, "UNKNOWN": 0}
        }

    def run(self):
        logger.info(f"Ingesting chemical properties from {self.file_path}")
        
        try:
            df = pd.read_excel(self.file_path)
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            raise

        self.stats["total_rows_read"] = len(df)
        
        # Renaissance Columns
        mapping = self.config.get('columns', {})
        df.rename(columns=mapping, inplace=True)
        
        unique_compounds = {} # canonical_id -> serialized_row
        
        for idx, row in df.iterrows():
            # Resolve Identity
            canonical_id, method, confidence = self.resolver.resolve(row)
            
            self.stats["resolution_methods"][method] += 1
            
            if method == "UNKNOWN":
                self.stats["skipped_rows"] += 1
                continue
                
            # Prepare Data Object
            compound_data = {
                "inchikey": row.get("inchikey") if pd.notna(row.get("inchikey")) else None,
                "inchi": row.get("inchi") if pd.notna(row.get("inchi")) else None,
                "smiles": row.get("smiles") if pd.notna(row.get("smiles")) else None,
                "cid": row.get("cid") if pd.notna(row.get("cid")) else None,
                "formula": row.get("formula") if pd.notna(row.get("formula")) else None,
                
                # Numeric Properties
                "mw": clean_float(row.get("mw")),
                "exact_mw": clean_float(row.get("exact_mw")),
                "logp": clean_float(row.get("logp")),
                "tpsa": clean_float(row.get("tpsa")),
                "hba": clean_int(row.get("hba")),
                "hbd": clean_int(row.get("hbd")),
                "rotb": clean_int(row.get("rotb")),
                "atom_count": clean_int(row.get("atom_count")), # mapped from ATOM
                "nring": clean_int(row.get("nring")),
                
                "property_source": "chemical_property.xlsx",
                "confidence": confidence,
                
                # Keep source_id for linking if needed, but primary ID is canonical
                "source_id": str(row.get("source_id"))
            }

            # If InChIKey is present, it MUST be the ID
            if method == "INCHIKEY":
                compound_data["compound_id"] = canonical_id # Identical to InChIKey
            elif method == "CID":
                compound_data["compound_id"] = canonical_id
            elif method == "SMILES":
                 compound_data["compound_id"] = canonical_id
            elif method == "SOURCE_ID":
                compound_data["compound_id"] = canonical_id

            # Deduplication: First wins or overwrite? 
            # "Never create duplicate compounds". 
            # We store in dict by canonical_id.
            if canonical_id not in unique_compounds:
                unique_compounds[canonical_id] = compound_data
            else:
                # Optional: Merge logic? For now, first wins as per "authoritative"
                pass

        self.stats["unique_compounds"] = len(unique_compounds)
        
        # Convert to DataFrame
        final_df = pd.DataFrame(list(unique_compounds.values()))
        
        output_csv = os.path.join(self.output_dir, "nodes_compound_properties.csv")
        final_df.to_csv(output_csv, index=False)
        logger.info(f"Saved {len(final_df)} compounds to {output_csv}")
        
        # Save Stats
        log_path = os.path.join(self.output_dir, "../logs/chemical_property_ingestion.json")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
            
        return final_df
