from backend.app.services.neo4j_service import Neo4jService

def debug_herbs():
    db = Neo4jService()
    try:
        print("--- Checking 'Gan Cao' (Glycyrrhiza) ---")
        # 1. Get the Herb node
        query_herb = "MATCH (h:Herb) WHERE h.name CONTAINS 'Glycyrrhizae' OR h.herbId = 'HERB_GLYCYRRHIZAE_RADIX_ET_RHIZOMA' RETURN h"
        with db.driver.session() as session:
            result = session.run(query_herb)
            for record in result:
                h = record['h']
                print(f"Herb Node: name='{h.get('name')}', scientificName='{h.get('scientificName')}', pinyin='{h.get('pinyin')}'")

        print("\n--- Checking HerbMaterial Relationships ---")
        # 2. Get HerbMaterial and check compounds
        query_material = """
        MATCH (hm:HerbMaterial) 
        WHERE hm.latin_name CONTAINS 'Glycyrrhizae' 
        OPTIONAL MATCH (hm)-[r:HAS_COMPOUND]->(c:Compound)
        RETURN hm.latin_name, count(c) as compound_count
        """
        with db.driver.session() as session:
             result = session.run(query_material)
             for record in result:
                 print(f"HerbMaterial: '{record['hm.latin_name']}' - Compounds: {record['compound_count']}")

        print("\n--- Checking Direct Herb Compounds (Fallback) ---")
        # 3. Check if Herb has direct CONTAINS relationship
        query_direct = """
        MATCH (h:Herb {herbId: 'HERB_GLYCYRRHIZAE_RADIX_ET_RHIZOMA'})
        OPTIONAL MATCH (h)-[:CONTAINS]->(c:Compound)
        RETURN count(c) as compound_count
        """
        with db.driver.session() as session:
             result = session.run(query_direct)
             record = result.single()
             print(f"Direct Herb->Compound count: {record['compound_count']}")

    finally:
        db.close()

if __name__ == "__main__":
    debug_herbs()
