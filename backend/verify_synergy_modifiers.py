import requests
import json

BASE_URL = "http://localhost:8000"

def test_synergy_with_modifiers():
    print("Testing Synergy with Modifiers...")
    
    # IDs for Turmeric and Ginger (using scientific name resolution or known IDs if possible)
    # Based on previous lookups:
    # Turmeric: HERB_1 (or similar? Need to check DB or use names if API supports it. 
    # The API expects IDs. Let's assume some IDs exist or search first.
    
    # 1. Search for Herbs to get IDs
    print("Searching for herbs...")
    try:
        r = requests.get(f"{BASE_URL}/herbs/")
        herbs = r.json()
        if len(herbs) < 2:
            print("Not enough herbs in DB to test synergy.")
            return
        
        id1 = herbs[0]['herbId']
        id2 = herbs[1]['herbId']
        name1 = herbs[0]['name']
        name2 = herbs[1]['name']
        print(f"Selected: {name1} ({id1}) + {name2} ({id2})")
        
        # 2. Test Synergy with Defaults
        payload_default = { "herbs": [id1, id2] }
        print("\nRequesting default synergy...")
        res_default = requests.post(f"{BASE_URL}/graph/synergy", json=payload_default)
        if res_default.status_code != 200:
            print(f"Error: {res_default.text}")
            return
        data_default = res_default.json()
        print("Default Result Keys:", data_default.keys())
        if 'vae_analysis' in data_default:
            print("Default Score:", data_default['vae_analysis'].get('synergy_score'))
        
        # 3. Test Synergy with High Alpha (Bio Weight)
        payload_alpha = { 
            "herbs": [id1, id2],
            "modifiers": { "alpha": 2.0, "beta": 0.5 }
        }
        print("\nRequesting High Alpha synergy...")
        res_alpha = requests.post(f"{BASE_URL}/graph/synergy", json=payload_alpha)
        data_alpha = res_alpha.json()
        if 'vae_analysis' in data_alpha:
            print("High Alpha Score:", data_alpha['vae_analysis'].get('synergy_score'))
            
        # 4. Test Synergy with High Beta (Redundancy Penalty)
        payload_beta = {
            "herbs": [id1, id2],
            "modifiers": { "alpha": 1.0, "beta": 1.0 }
        }
        print("\nRequesting High Beta synergy...")
        res_beta = requests.post(f"{BASE_URL}/graph/synergy", json=payload_beta)
        data_beta = res_beta.json()
        if 'vae_analysis' in data_beta:
            print("High Beta Score:", data_beta['vae_analysis'].get('synergy_score'))

    except Exception as e:
        print(f"Test failed: {e}")

def test_herb_expansion():
    print("\nTesting Herb Expansion...")
    try:
        r = requests.get(f"{BASE_URL}/herbs/")
        herbs = r.json()
        if not herbs: return
        
        target_id = herbs[0]['herbId']
        print(f"Expanding Herb: {herbs[0]['name']} ({target_id})")
        
        res = requests.get(f"{BASE_URL}/graph/expand/herb/{target_id}")
        data = res.json()
        
        print(f"Expanded Nodes: {len(data.get('nodes', []))}")
        print(f"Expanded Edges: {len(data.get('edges', []))}")
        
        # Check for compounds
        compounds = [n for n in data.get('nodes', []) if n['type'] == 'compound']
        print(f"Compounds found: {len(compounds)}")
        
    except Exception as e:
         print(f"Expansion test failed: {e}")

if __name__ == "__main__":
    test_synergy_with_modifiers()
    test_herb_expansion()
