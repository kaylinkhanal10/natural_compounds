from backend.app.services.neo4j_service import Neo4jService
import sys
import os

sys.path.append(os.getcwd())

def count_nodes():
    try:
        neo = Neo4jService()
        with neo.driver.session() as session:
            res = session.run("MATCH (n) RETURN count(n) as count")
            count = res.single()['count']
            print(f"Total Nodes: {count}")
            
            res = session.run("MATCH (h:Herb) RETURN count(h) as count")
            print(f"Herbs: {res.single()['count']}")
            
            res = session.run("MATCH (c:Compound) RETURN count(c) as count")
            print(f"Compounds: {res.single()['count']}")
        neo.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    count_nodes()
