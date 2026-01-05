from neo4j import GraphDatabase
import os

uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"

driver = GraphDatabase.driver(uri, auth=(user, password))

HERB_NAME_MAP = {
    "Curcuma longa": "Curcumae Longae Rhizoma",              # Turmeric
    "Piper longum": "Piperis Longi Fructus",                 # Pippali
    "Zingiber officinale": "Zingiberis Rhizoma",             # Ginger
    "Phyllanthus emblica": "Phyllanthi Fructus",             # Amla
    "Tinospora cordifolia": "Tinosporae Radix",              # Guduchi
    "Glycyrrhiza glabra": "Glycyrrhizae Radix Et Rhizoma"    # Licorice
}

herbs = ["HERB_TURMERIC", "HERB_PIPPALI"]

query = """
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

try:
    with driver.session() as session:
        print(f"Checking herbs: {herbs}")
        
        # 1. Check Herb Nodes
        q1 = "MATCH (h:Herb) WHERE h.herbId IN $herbs RETURN h.name, h.scientificName, h.herbId"
        res1 = session.run(q1, herbs=herbs).data()
        print(f"1. Found Herbs: {res1}")
        
        # 2. Check Compounds (using mapping logic)
        q2 = """
        MATCH (h:Herb) WHERE h.herbId IN $herbs
        WITH h, $map[h.scientificName] as mapped_name
        MATCH (hm:HerbMaterial) 
        WHERE hm.latin_name = h.scientificName OR (mapped_name IS NOT NULL AND hm.latin_name = mapped_name)
        MATCH (hm)-[:HAS_COMPOUND]->(c:Compound)
        RETURN h.name, count(c) as compound_count
        """
        res2 = session.run(q2, herbs=herbs, map=HERB_NAME_MAP).data()
        print(f"2. Compound Counts: {res2}")
        
        # 3. Check Targets
        q3 = """
        MATCH (h:Herb) WHERE h.herbId IN $herbs
        WITH h, $map[h.scientificName] as mapped_name
        MATCH (hm:HerbMaterial) 
        WHERE hm.latin_name = h.scientificName OR (mapped_name IS NOT NULL AND hm.latin_name = mapped_name)
        MATCH (hm)-[:HAS_COMPOUND]->(c:Compound)-[:TARGETS]->(p:Protein)
        RETURN h.name, count(p) as target_count
        """
        res3 = session.run(q3, herbs=herbs, map=HERB_NAME_MAP).data()
        print(f"3. Target Counts: {res3}")
        
        # 5. Check Compound Properties
        print("Checking Compound Properties...")
        q_props = """
        MATCH (c:Compound) 
        RETURN c.name, keys(c) as properties, c.inchiKey, c.smiles
        LIMIT 1
        """
        res_props = session.run(q_props).data()
        print(f"Compound Props: {res_props}")

except Exception as e:
    print(f"QUERY FAILED: {e}")
finally:
    driver.close()
