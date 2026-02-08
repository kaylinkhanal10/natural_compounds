from app.services.neo4j_service import Neo4jService

def align_names():
    db = Neo4jService()
    updates = [
        # ID, Current Scientific, Target Latin (from HerbMaterial)
        ('HERB_TURMERIC', 'Curcuma longa', 'Curcumae Longae Rhizoma'),
        ('HERB_GINSENG', 'Panax ginseng', 'Ginseng Radix Et Rhizoma'), # Guessing, check first?
        ('HERB_LICORICE', 'Glycyrrhiza glabra', 'Glycyrrhizae Radix Et Rhizoma') # Guessing
    ]
    
    try:
        with db.driver.session() as session:
            # First, verify the target latin names exist
            print("Verifying target names...")
            for _, _, target in updates:
                res = session.run("MATCH (hm:HerbMaterial {latin_name: $name}) RETURN count(hm) as c", name=target).single()
                if res and res['c'] > 0:
                    print(f"Target '{target}' EXISTS.")
                else:
                    print(f"Target '{target}' NOT FOUND. Finding partial match...")
                    parts = target.split(' ')[0]
                    res_partial = session.run("MATCH (hm:HerbMaterial) WHERE hm.latin_name CONTAINS $p RETURN hm.latin_name LIMIT 3", p=parts)
                    for r in res_partial:
                        print(f"  Did you mean: {r['hm.latin_name']}?")

    finally:
        db.close()

if __name__ == "__main__":
    align_names()
