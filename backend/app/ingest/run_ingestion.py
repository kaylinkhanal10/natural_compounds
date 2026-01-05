import pandas as pd
import yaml
import os
import logging
import json
from .normalize import simple_title_case, clean_float, clean_int
from .chemical_property_ingest import ChemicalPropertyIngest
from .compound_resolver import CompoundResolver

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IngestionPipeline:
    def __init__(self, raw_data_dir, output_dir, config_path):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.stats = {}

    def read_excel(self, file_key):
        cfg = self.config['files'][file_key]
        path = os.path.join(self.raw_data_dir, cfg['filename'])
        logger.info(f"Reading {path}...")
        try:
            df = pd.read_excel(path)
            # Rename columns
            df.rename(columns=cfg['columns'], inplace=True)
            self.stats[file_key] = {"rows": len(df)}
            return df
        except Exception as e:
            logger.error(f"Failed to read {file_key}: {e}")
            raise

    def run(self):
        logger.info("Starting Ingestion...")
        
        # 1. Chemical Properties (Compounds) - Master List
        # New Module Integration
        chem_ingest = ChemicalPropertyIngest(
            file_path=os.path.join(self.raw_data_dir, self.config['files']['chemical_property']['filename']),
            config=self.config['files']['chemical_property'],
            output_dir=self.output_dir
        )
        # This generates nodes_compound_properties.csv
        df_final_compounds = chem_ingest.run()
        
        # Create ID Map for other files
        # We need to map source_id -> canonical_id to update foreign keys in other sheets.
        # Since ChemicalPropertyIngest handles resolution, we can rebuild the map by running the resolver over the raw file again.
        # Ideally, ChemicalPropertyIngest would return this map, but for separation of concerns, we can re-derive it.
        
        logger.info("Building ID map for linking...")
        raw_comp_path = os.path.join(self.raw_data_dir, self.config['files']['chemical_property']['filename'])
        df_raw = pd.read_excel(raw_comp_path)
        df_raw.rename(columns=self.config['files']['chemical_property']['columns'], inplace=True)
        
        id_map = {}
        resolver = CompoundResolver()
        
        for idx, row in df_raw.iterrows():
            canonical, method, _ = resolver.resolve(row)
            src = str(row.get('source_id'))
            
            # If we resolved it, map source -> canonical
            if method != "UNKNOWN":
                 id_map[src] = canonical
            else:
                 # If unknown, keep original (orphan or fallback)
                 id_map[src] = src

        logger.info(f"Generated ID map with {len(id_map)} entries")

        # 2. Herbs (Medicinal Material)
        df_herbs = self.read_excel('medicinal_material')
        df_herbs['latin_name'] = df_herbs['latin_name'].apply(simple_title_case)
        df_herbs.drop_duplicates(subset=['latin_name'], inplace=True)
        df_herbs.to_csv(os.path.join(self.output_dir, 'nodes_herb_material.csv'), index=False)
        
        # 3. Medicinal Compound (Herb -> Compound)
        df_herb_cmp = self.read_excel('medicinal_compound')
        df_herb_cmp['herb_latin'] = df_herb_cmp['herb_latin'].apply(simple_title_case)
        df_herb_cmp['compound_id'] = df_herb_cmp['compound_id'].astype(str)
        # Remap IDs
        df_herb_cmp['compound_id'] = df_herb_cmp['compound_id'].map(id_map).fillna(df_herb_cmp['compound_id']) 
        
        df_herb_cmp.to_csv(os.path.join(self.output_dir, 'edges_herb_compound.csv'), index=False)

        # 4. Chemical Protein (Compound -> Protein)
        df_cmp_prot = self.read_excel('chemical_protein')
        df_cmp_prot['compound_id'] = df_cmp_prot['compound_id'].astype(str)
        # Remap IDs
        df_cmp_prot['compound_id'] = df_cmp_prot['compound_id'].map(id_map).fillna(df_cmp_prot['compound_id'])
        
        df_cmp_prot['protein_id'] = df_cmp_prot['protein_id'].astype(str)
        
        # Extract Unique Proteins
        proteins = df_cmp_prot[['protein_id', 'protein_name']].drop_duplicates(subset=['protein_id'])
        proteins.to_csv(os.path.join(self.output_dir, 'nodes_protein.csv'), index=False)
        
        df_cmp_prot.to_csv(os.path.join(self.output_dir, 'edges_compound_protein.csv'), index=False)

        # 5. Protein Disease (Protein -> Disease)
        df_prot_dis = self.read_excel('protein_disease')
        df_prot_dis['protein_id'] = df_prot_dis['protein_id'].astype(str)
        df_prot_dis['disease_id'] = df_prot_dis['disease_id'].astype(str)
        
        # Extract Unique Diseases
        diseases = df_prot_dis[['disease_id', 'disease_name']].drop_duplicates(subset=['disease_id'])
        diseases.to_csv(os.path.join(self.output_dir, 'nodes_disease.csv'), index=False)
        
        df_prot_dis.to_csv(os.path.join(self.output_dir, 'edges_protein_disease.csv'), index=False)

        # 6. Prescription (Formula -> Herb)
        df_rx = self.read_excel('prescription')
        df_rx['ingredient_latin'] = df_rx['ingredient_latin'].apply(simple_title_case)
        df_rx['formula_name'] = df_rx['formula_name'].apply(simple_title_case)
        
        # Unique Formulas
        formulas = df_rx[['formula_name', 'source_book']].drop_duplicates(subset=['formula_name'])
        formulas.to_csv(os.path.join(self.output_dir, 'nodes_formula.csv'), index=False)
        
        df_rx.to_csv(os.path.join(self.output_dir, 'edges_formula_herb.csv'), index=False)

        # Save stats
        with open(os.path.join(self.output_dir, '../logs/ingestion_summary.json'), 'a') as f: # Append or separate file?
            # chemical_property_ingest wrote its own logs. We can append existing stats.
             json.dump(self.stats, f, indent=2)
            
        logger.info("Ingestion Complete. CSVs generated in staging.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # RAW_DIR = os.path.join(BASE_DIR, "../../raw_data") 
    
    pipe = IngestionPipeline(
        raw_data_dir="/home/peridot/discovery/natural-medicine-ai/raw_data", 
        output_dir="/home/peridot/discovery/natural-medicine-ai/backend/app/data/staging",
        config_path="/home/peridot/discovery/natural-medicine-ai/backend/app/ingest/column_map.yaml"
    )
    pipe.run()
