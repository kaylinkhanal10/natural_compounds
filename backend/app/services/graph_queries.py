
from .neo4j_service import Neo4jService

class GraphQueryService:
    def __init__(self):
        self.db = Neo4jService()

    def close(self):
        self.db.close()

    def get_formulas(self, limit=50):
        query = """
        MATCH (f:Formula)
        OPTIONAL MATCH (f)-[r:HAS_INGREDIENT]->(h:HerbMaterial)
        RETURN f.name as name, f.source_book as source, count(h) as ingredient_count
        ORDER BY f.name
        LIMIT $limit
        """
        with self.db.driver.session() as session:
            result = session.run(query, limit=limit)
            return [record.data() for record in result]

    def get_formula_details(self, name):
        query = """
        MATCH (f:Formula {name: $name})
        OPTIONAL MATCH (f)-[r:HAS_INGREDIENT]->(h:HerbMaterial)
        // Expand mechanistically
        OPTIONAL MATCH (h)-[:HAS_COMPOUND]->(c:Compound)-[:TARGETS]->(p:Protein)-[:ASSOCIATED_WITH]->(d:Disease)
        RETURN f, 
               collect(DISTINCT {name: h.latin_name, dosage: r.dosage, unit: r.unit, role: r.seq}) as ingredients,
               count(DISTINCT c) as compound_count,
               count(DISTINCT p) as protein_count,
               collect(DISTINCT d.name)[..10] as top_diseases
        """
        with self.db.driver.session() as session:
            result = session.run(query, name=name)
            record = result.single()
            if record and record['f']:
                data = record['f']
                data['ingredients'] = record['ingredients']
                data['mechanistics'] = {
                    "compounds": record['compound_count'],
                    "targets": record['protein_count'],
                    "diseases": record['top_diseases']
                }
                return data
            return None

    def get_diseases(self, limit=50):
        query = """
        MATCH (d:Disease)
        RETURN d.id as id, d.name as name
        ORDER BY d.name
        LIMIT $limit
        """
        with self.db.driver.session() as session:
            result = session.run(query, limit=limit)
            return [record.data() for record in result]

    def get_disease_details(self, disease_id):
        query = """
        MATCH (d:Disease {id: $id})
        OPTIONAL MATCH (p:Protein)-[:ASSOCIATED_WITH]->(d)
        OPTIONAL MATCH (c:Compound)-[:TARGETS]->(p)
        OPTIONAL MATCH (h:HerbMaterial)-[:HAS_COMPOUND]->(c)
        RETURN d, 
               collect(DISTINCT p.name) as proteins,
               collect(DISTINCT c.name) as compounds,
               collect(DISTINCT h.latin_name) as herbs
        """
        with self.db.driver.session() as session:
            result = session.run(query, id=disease_id)
            record = result.single()
            if record and record['d']:
                data = record['d']
                data['proteins'] = record['proteins']
                data['related_compounds_count'] = len(record['compounds'])
                data['related_herbs'] = record['herbs'][:20] # Limit response
                return data
            return None

    def get_herb_details_expanded(self, herb_id):
        # Merge logic for Phase 1 (Herb) and Phase 2 (HerbMaterial)
        # Assuming ID matches or names match.
        # Phase 1: Herb {herbId: 'HERB_AMLA', name: 'Amla'}
        # Phase 2: HerbMaterial {latin_name: 'Phyllanthus emblica'} (Scientific)
        # We need to bridge them.
        # For now, let's assume UI passes 'Amla' or 'Phyllanthus emblica'.
        # If ID passed is 'HERB_AMLA', look up Phase 1 node.
        query = """
        MATCH (h:Herb {herbId: $herb_id})
        // Try to link to HerbMaterial via name (fuzzy) or manual map?
        // Let's rely on name matching or just return what we have.
        // Phase 2: HerbMaterial key is latin_name. Phase 1 Herb has scientificName.
        OPTIONAL MATCH (hm:HerbMaterial) WHERE hm.latin_name = h.scientificName
        // Expand
        OPTIONAL MATCH (hm)-[:HAS_COMPOUND]->(c:Compound)
        OPTIONAL MATCH (c)-[:TARGETS]->(p:Protein)-[:ASSOCIATED_WITH]->(d:Disease)
        RETURN h, 
               collect(DISTINCT c {
                   compoundId: c.compoundId,
                   name: c.name, 
                   inchikey: c.inchikey, 
                   mw: c.mw, 
                   logp: c.logp,
                   tpsa: c.tpsa,
                   hba: c.hba,
                   hbd: c.hbd,
                   rotb: c.rotb
               }) as chemical_compounds,
               collect(DISTINCT p.name) as targets,
               collect(DISTINCT d.name) as diseases
        """
        with self.db.driver.session() as session:
            result = session.run(query, herb_id=herb_id)
            record = result.single()
            if record and record['h']:
                data = record['h']
                data['extended_compounds'] = record['chemical_compounds']
                data['targets'] = record['targets']
                data['diseases'] = record['diseases']
                return data
            return None

    def analyze_combination_expanded(self, herbs):
        # herbs = list of herb IDs or names
        # Need to Map Phase 1 IDs to Phase 2 HerbMaterials
        
        # Explicit Mapping for MVP Herbs
        HERB_NAME_MAP = {
            "Curcuma longa": "Curcumae Longae Rhizoma",              # Turmeric
            "Piper longum": "Piperis Longi Fructus",                 # Pippali
            "Zingiber officinale": "Zingiberis Rhizoma",             # Ginger
            "Phyllanthus emblica": "Phyllanthi Fructus",             # Amla
            "Tinospora cordifolia": "Tinosporae Radix",              # Guduchi
            "Glycyrrhiza glabra": "Glycyrrhizae Radix Et Rhizoma"    # Licorice
        }

        query = """
        MATCH (h:Herb) WHERE h.herbId IN $herbs
        WITH h, $map[h.scientificName] as mapped_name
        MATCH (hm:HerbMaterial) 
        WHERE hm.latin_name = h.scientificName OR (mapped_name IS NOT NULL AND hm.latin_name = mapped_name)
        WITH collect(hm) as materials
        UNWIND materials as m
        MATCH (m)-[:HAS_COMPOUND]->(c:Compound)
        MATCH (c)-[:TARGETS]->(p:Protein)
        OPTIONAL MATCH (p)-[:ASSOCIATED_WITH]->(d:Disease)
        
        RETURN collect(DISTINCT c.name) as compounds,
               collect(DISTINCT p.name) as proteins,
               collect(DISTINCT d.name) as diseases,
               // Find shared proteins
               p.name as protein_name, count(DISTINCT m) as herb_support
        """
        # This aggregation is tricky in one query.
        # Split:
        # 1. Union of coverage
        # 2. Shared targets
        
        q_shared = """
        MATCH (h:Herb) WHERE h.herbId IN $herbs
        WITH h, $map[h.scientificName] as mapped_name
        MATCH (hm:HerbMaterial) 
        WHERE hm.latin_name = h.scientificName OR (mapped_name IS NOT NULL AND hm.latin_name = mapped_name)
        MATCH (hm)-[:HAS_COMPOUND]->(c:Compound)-[:TARGETS]->(p:Protein)
        WITH p, count(DISTINCT hm) as h_count, collect(DISTINCT h.name) as sources
        WHERE h_count > 1
        RETURN p.name as name, h_count as count, sources
        ORDER BY count DESC
        """
        
        q_union = """
        MATCH (h:Herb) WHERE h.herbId IN $herbs
        WITH h, $map[h.scientificName] as mapped_name
        MATCH (hm:HerbMaterial) 
        WHERE hm.latin_name = h.scientificName OR (mapped_name IS NOT NULL AND hm.latin_name = mapped_name)
        MATCH (hm)-[:HAS_COMPOUND]->(c:Compound)
        OPTIONAL MATCH (c)-[:TARGETS]->(p:Protein)
        OPTIONAL MATCH (p)-[:ASSOCIATED_WITH]->(d:Disease)
        RETURN count(DISTINCT c) as comp_count, count(DISTINCT p) as prot_count, collect(DISTINCT d.name) as diseases
        """
        
        with self.db.driver.session() as session:
             shared_res = session.run(q_shared, herbs=herbs, map=HERB_NAME_MAP)
             shared = [r.data() for r in shared_res]
             
             union_res = session.run(q_union, herbs=herbs, map=HERB_NAME_MAP)
             union = union_res.single().data()
             
             # Fetch compound details for feasibility check
             q_feasibility = """
             MATCH (h:Herb) WHERE h.herbId IN $herbs
             WITH h, $map[h.scientificName] as mapped_name
             MATCH (hm:HerbMaterial) 
             WHERE hm.latin_name = h.scientificName OR (mapped_name IS NOT NULL AND hm.latin_name = mapped_name)
             MATCH (hm)-[:HAS_COMPOUND]->(c:Compound)
             RETURN DISTINCT c.name as name, c.mw as mw, c.tpsa as tpsa, c.logp as logp, h.name as herb, c.inchikey as inchikey
             """
             feas_res = session.run(q_feasibility, herbs=herbs, map=HERB_NAME_MAP)
             feasibility = [r.data() for r in feas_res]
             
             return {
                 "shared_targets": shared,
                 "counts": union,
                 "chemical_feasibility": feasibility
             }
