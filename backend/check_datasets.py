from neo4j import GraphDatabase

# Configuration
uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"

driver = GraphDatabase.driver(uri, auth=(user, password))

def run_deep_inspection():
    queries = {
        "Compound Properties": "MATCH (n:Compound) RETURN keys(n) as props, n LIMIT 1",
        "Herb Properties": "MATCH (n:Herb) RETURN keys(n) as props, n LIMIT 1",
        "Protein Properties": "MATCH (n:Protein) RETURN keys(n) as props, n LIMIT 1",
        "Disease Properties": "MATCH (n:Disease) RETURN keys(n) as props, n LIMIT 1",
        "HerbMaterial Properties": "MATCH (n:HerbMaterial) RETURN keys(n) as props, n LIMIT 1",
    }

    try:
        with driver.session() as session:
            print("--- Deep Property Inspection ---")
            for name, query in queries.items():
                print(f"\n{name}:")
                try:
                    results = session.run(query).data()
                    for row in results:
                        # Print the node content directly to see key-values
                        print(row)
                except Exception as e:
                    print(f"Error: {e}")

    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    run_deep_inspection()
