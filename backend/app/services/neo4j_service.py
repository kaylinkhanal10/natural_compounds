from neo4j import GraphDatabase
import os

class Neo4jService:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_all_herbs(self):
        query = """
        MATCH (h:Herb)
        RETURN h.herbId as herbId, h.name as name, h.scientificName as scientificName, 
               h.description as description, h.sanskritName as sanskritName
        ORDER BY h.name
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

    def get_herb_details(self, herb_id: str):
        query = """
        MATCH (h:Herb {herbId: $herb_id})
        OPTIONAL MATCH (h)-[:CONTAINS]->(c:Compound)
        OPTIONAL MATCH (h)-[:SUPPORTED_BY]->(e:Evidence)
        OPTIONAL MATCH (h)-[:CONTAINS]->(:Compound)-[:MODULATES]->(be:BiologicalEffect)
        RETURN h, collect(DISTINCT c) as compounds, collect(DISTINCT be) as effects, collect(DISTINCT e) as evidence
        """
        with self.driver.session() as session:
            result = session.run(query, herb_id=herb_id)
            record = result.single()
            if record:
                data = record['h']
                data['compounds'] = [c for c in record['compounds']]
                data['effects'] = [e for e in record['effects']]
                data['evidence'] = [e for e in record['evidence']]
                return data
            return None

    def find_shared_effects(self, herb_ids: list):
        query = """
        MATCH (h:Herb)-[:CONTAINS]->(:Compound)-[:MODULATES]->(e:BiologicalEffect)
        WHERE h.herbId IN $herb_ids
        WITH e, count(DISTINCT h) as herb_count, collect(DISTINCT h.name) as herb_names
        WHERE herb_count > 1
        RETURN e, herb_names
        """
        with self.driver.session() as session:
            result = session.run(query, herb_ids=herb_ids, total_herbs=len(herb_ids))
            return [record.data() for record in result]
    
    def find_interactions(self, herb_ids: list):
        # Placeholder logic for interactions:
        # Check if any herb in the list interacts solely with another herb in the list
        # OR if compounds interact.
        # Minimal demo version.
        query = """
        MATCH (h1:Herb)-[r:HAS_INTERACTION]->(i:Interaction)<-[r2:HAS_INTERACTION]-(h2:Herb)
        WHERE h1.herbId IN $herb_ids AND h2.herbId IN $herb_ids
        RETURN h1.name, h2.name, i.description, i.severity
        """
        # Also check compound interactions
        query_compounds = """
        MATCH (h1:Herb)-[:CONTAINS]->(c1:Compound)-[r:ENHANCES_ABSORPTION]->(c2:Compound)<-[:CONTAINS]-(h2:Herb)
        WHERE h1.herbId IN $herb_ids AND h2.herbId IN $herb_ids AND h1 <> h2
        RETURN h1.name, h2.name, c1.name, c2.name, r.mechanism
        """
        interactions = []
        with self.driver.session() as session:
            # Direct herb interaction (not in my dummy csv yet but good for logic)
            # res = session.run(query, herb_ids=herb_ids)
            
            # Compound interaction (Enhances absorption)
            res_comp = session.run(query_compounds, herb_ids=herb_ids)
            for record in res_comp:
                interactions.append({
                    "type": "SYNERGY", 
                    "description": f"{record['h1.name']} ({record['c1.name']}) enhances absorption of {record['h2.name']} ({record['c2.name']}) via {record['r.mechanism']}"
                })
        return interactions

