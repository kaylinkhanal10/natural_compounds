from app.services.neo4j_service import Neo4jService
import sys

def check_turmeric():
    db = Neo4jService()
    try:
        print("Checking for Herb node...")
        with db.driver.session() as session:
            # Check Herb
            res = session.run("MATCH (h:Herb {herbId: 'HERB_TURMERIC'}) RETURN h.name, h.scientificName").single()
            if res:
                print(f"Found Herb: {res['h.name']}, Scientific: {res['h.scientificName']}")
                sci_name = res['h.scientificName']
            else:
                print("Herb HERB_TURMERIC not found!")
                return

            # Check HerbMaterial
            print(f"Checking for HerbMaterial with latin_name='{sci_name}'...")
            res_hm = session.run("MATCH (hm:HerbMaterial {latin_name: $name}) RETURN hm.latin_name, count(hm) as c", name=sci_name).single()
            if res_hm and res_hm['c'] > 0:
                print(f"Found HerbMaterial: {res_hm['hm.latin_name']}")
            else:
                print(f"HerbMaterial with latin_name='{sci_name}' NOT FOUND.")
                
                # Check for any HerbMaterial to see what they look like
                print("Sample HerbMaterials:")
                sample = session.run("MATCH (hm:HerbMaterial) RETURN hm.latin_name LIMIT 5")
                for r in sample:
                    print(f" - {r['hm.latin_name']}")

            # Check Compounds
            print("Checking linked compounds...")
            query = """
            MATCH (hm:HerbMaterial {latin_name: $name})-[:HAS_COMPOUND]->(c:Compound)
            RETURN count(c) as count
            """
            res_c = session.run(query, name=sci_name).single()
            print(f"Compound Count: {res_c['count'] if res_c else 0}")

    finally:
        db.close()

if __name__ == "__main__":
    check_turmeric()
