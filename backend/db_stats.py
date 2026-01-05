from neo4j import GraphDatabase
import os

uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"

driver = GraphDatabase.driver(uri, auth=(user, password))

try:
    with driver.session() as session:
        print("--- Node Counts ---")
        query_nodes = """
        MATCH (n)
        RETURN labels(n) as label, count(n) as count
        ORDER BY count DESC
        """
        result = session.run(query_nodes)
        for record in result:
            print(f"{record['label']}: {record['count']}")

        print("\n--- Relationship Counts ---")
        query_rels = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(r) as count
        ORDER BY count DESC
        """
        result = session.run(query_rels)
        for record in result:
            print(f"{record['type']}: {record['count']}")

except Exception as e:
    print(f"QUERY FAILED: {e}")
finally:
    driver.close()
