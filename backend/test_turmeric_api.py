from app.services.graph_queries import GraphQueryService
import json

def test_api_logic():
    service = GraphQueryService()
    try:
        print("Calling get_herb_details_expanded('HERB_TURMERIC')...")
        data = service.get_herb_details_expanded('HERB_TURMERIC')
        
        if data:
            print(f"Herb Name: {data.get('name')}")
            print(f"Scientific: {data.get('scientificName')}")
            compounds = data.get('extended_compounds', [])
            print(f"Extended Compounds Count: {len(compounds)}")
            if len(compounds) > 0:
                print(f"Sample Compound: {compounds[0]['name']}")
        else:
            print("Returned None!")

    finally:
        service.close()

if __name__ == "__main__":
    test_api_logic()
