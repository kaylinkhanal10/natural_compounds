
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
        OPTIONAL MATCH (h)-[:HAS_COMPOUND]->(c:Compound)
        OPTIONAL MATCH (c)-[:TARGETS]->(p:Protein)-[:ASSOCIATED_WITH]->(d:Disease)
        RETURN h, 
               collect(DISTINCT c {
                   compoundId: c.compoundId,
                   name: c.name, 
                   inchikey: c.inchikey, 
                   smiles: c.smiles,
                   targets: [(c)-[:TARGETS]->(t:Protein) | t.name],
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
                data = dict(record['h'])
                data['extended_compounds'] = record['chemical_compounds']
                data['targets'] = record['targets']
                data['diseases'] = record['diseases']
                return data
            return None

    def analyze_combination_expanded(self, herbs, modifiers=None):
        """
        Delegates to ReasoningEngine for neuro-symbolic analysis.
        """
        from .reasoning_engine import ReasoningEngine
        engine = ReasoningEngine()
        try:
            return engine.analyze_combination(herbs, modifiers)
        finally:
            engine.close()

    def expand_node_generic(self, node_type: str, node_id: str):
        """
        Generic expansion for graph exploration.
        Returns immediate neighbors.
        """
        # 1. Determine label and ID property based on type
        label = "Herb"
        id_prop = "herbId"
        
        if node_type == "herb":
            label = "Herb"
            id_prop = "herbId"
        elif node_type == "compound":
            label = "Compound"
            id_prop = "compoundId" 
        elif node_type == "target":
            label = "Protein"
            id_prop = "id" # Assuming Protein has id or name. Check schema.
        elif node_type == "disease":
            label = "Disease"
            id_prop = "id"
            
        # Simplified query to get neighbors
        # We match by elementId(n) if node_id look like an internal ID, or by property.
        # Ideally, we should unify ID strategy. For now, assume property match.
        
        # Enhanced query for Herb expansion to include compounds via HerbMaterial bridge
        if node_type == "herb":
             query = f"""
             MATCH (n:{label}) WHERE n.{id_prop} = $id OR n.name = $id
             WITH n
             // Optional bridge to HerbMaterial
             OPTIONAL MATCH (hm:HerbMaterial) WHERE hm.latin_name = n.scientificName
             WITH n, hm
             // Expand from n directly OR from hm
             MATCH (start)-[r]-(m)
             WHERE start = n OR start = hm
             RETURN start as n, r, m, labels(m) as m_labels, type(r) as r_type
             LIMIT 50
             """
        else:
             query = f"""
             MATCH (n:{label}) WHERE n.{id_prop} = $id OR n.name = $id
             MATCH (n)-[r]-(m)
             RETURN n, r, m, labels(m) as m_labels, type(r) as r_type
             LIMIT 50
             """
        
        with self.db.driver.session() as session:
            result = session.run(query, id=node_id)
            nodes = {}
            edges = []
            
            for record in result:
                n = record['n']
                m = record['m']
                r = record['r']
                
                # Helper to format node
                def format_node(node, labels):
                    return {
                        "id": node.get(id_prop, node.get("id", node.get("name", str(node.id)))), # Fallback
                        "type": list(labels)[0].lower(), 
                        "data": dict(node)
                    }

                # Source Node
                n_id = n.get(id_prop, n.get("id", n.get("name", str(n.id))))
                if n_id not in nodes:
                    nodes[n_id] = format_node(n, [label]) # We know the label of n
                    
                # Target Node (Neighbor)
                m_id = m.get("herbId", m.get("compoundId", m.get("id", m.get("name", str(m.id)))))
                if m_id not in nodes:
                    nodes[m_id] = format_node(m, record['m_labels'])
                    
                # Edge
                edges.append({
                    "id": str(r.id),
                    "source": n_id,
                    "target": m_id,
                    "label": record['r_type']
                })
                
            return {
                "nodes": list(nodes.values()),
                "edges": edges
            }
