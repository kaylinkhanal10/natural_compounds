from app.services.neo4j_service import Neo4jService

def update_herbs():
    db = Neo4jService()
    updates = [
        ('HERB_TURMERIC', 'Curcumae Longae Rhizoma'),
        ('HERB_GINSENG', 'Ginseng Radix'),
        ('HERB_LICORICE', 'Glycyrrhizae Radix Et Rhizoma')
    ]
    
    try:
        with db.driver.session() as session:
            for hid, new_name in updates:
                print(f"Updating {hid} to scientificName='{new_name}'...")
                session.run("MATCH (h:Herb {herbId: $hid}) SET h.scientificName = $name", hid=hid, name=new_name)
                print("Done.")
            
            # Verify Turmeric again
            print("Verifying Turmeric connection...")
            res = session.run("""
                MATCH (h:Herb {herbId: 'HERB_TURMERIC'})
                MATCH (hm:HerbMaterial {latin_name: h.scientificName})
                MATCH (hm)-[:HAS_COMPOUND]->(c:Compound)
                RETURN count(c) as count
            """).single()
            print(f"Turmeric Linked Compounds: {res['count'] if res else 0}")

    finally:
        db.close()

if __name__ == "__main__":
    update_herbs()
