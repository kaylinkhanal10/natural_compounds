from app.services.neo4j_service import Neo4jService

def find_turmeric():
    db = Neo4jService()
    try:
        with db.driver.session() as session:
            print("Searching for HerbMaterial containing 'Curcuma'...")
            res = session.run("MATCH (hm:HerbMaterial) WHERE hm.latin_name CONTAINS 'Curcuma' RETURN hm.latin_name")
            found = False
            for r in res:
                print(f" - {r['hm.latin_name']}")
                found = True
            
            if not found:
                print("No 'Curcuma' found. Trying 'Longa'...")
                res = session.run("MATCH (hm:HerbMaterial) WHERE hm.latin_name CONTAINS 'Longa' RETURN hm.latin_name")
                for r in res:
                    print(f" - {r['hm.latin_name']}")

    finally:
        db.close()

if __name__ == "__main__":
    find_turmeric()
