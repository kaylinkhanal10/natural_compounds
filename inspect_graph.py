from backend.app.services.neo4j_service import Neo4jService
import sys
import os

sys.path.append(os.getcwd())

def inspect_compound():
    neo = Neo4jService()
    with neo.driver.session() as session:
        res = session.run("MATCH (c:Compound) RETURN c LIMIT 1")
        record = res.single()
        if record:
            print(record['c'].keys())
            print(record['c'])
        else:
            print("No compounds found")
    neo.close()

if __name__ == "__main__":
    inspect_compound()
