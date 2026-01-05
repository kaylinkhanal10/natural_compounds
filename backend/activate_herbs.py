from neo4j import GraphDatabase
import re

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

def sanitize(name):
    # Create a valid ID from latin name (e.g. "Panax ginseng" -> "HERB_PANAX_GINSENG")
    clean = re.sub(r'[^a-zA-Z0-9]', '_', name).upper()
    return f"HERB_{clean}"

migration_query = """
MATCH (hm:HerbMaterial)
WHERE NOT EXISTS {
    MATCH (h:Herb)
    WHERE h.scientificName = hm.latin_name
}
RETURN hm.latin_name as latin_name
"""

create_query = """
MERGE (h:Herb {herbId: $herbId})
SET h.name = $name,
    h.scientificName = $scientificName,
    h.description = "Activated from full catalog.",
    h.sanskritName = $name // Default to latin if unknown
"""

with driver.session() as session:
    # 1. Fetch Candidates
    print("Scanning for hidden herbs...")
    result = session.run(migration_query)
    candidates = [r["latin_name"] for r in result]
    
    print(f"Found {len(candidates)} hidden herbs to activate.")
    
    # 2. Activate
    count = 0
    for latin in candidates:
        herb_id = sanitize(latin)
        # Use simple name strategy: First word or full latin
        # e.g. "Panax ginseng" -> "Panax ginseng"
        
        session.run(create_query, 
                    herbId=herb_id, 
                    name=latin, 
                    scientificName=latin)
        count += 1
        if count % 50 == 0:
            print(f"Activated {count}...")

    print(f"SUCCESS: Activated {count} new herbs.")
    print("The UI should now show the full list.")
