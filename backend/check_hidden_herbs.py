from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

query = """
MATCH (hm:HerbMaterial)
WHERE NOT EXISTS {
    MATCH (h:Herb)
    WHERE h.scientificName = hm.latin_name
}
RETURN hm.latin_name as name, count { (hm)-[:HAS_COMPOUND]->() } as compound_count
ORDER BY compound_count DESC
LIMIT 10
"""

with driver.session() as session:
    result = session.run(query)
    print("--- Hidden Herbs (Data Present, UI Hidden) ---")
    for r in result:
        print(f"Herb: {r['name']} | Compounds: {r['compound_count']}")
