# test_endpoints.py
import requests
import json

def test_all_endpoints():
    base_url = "http://localhost:5001"
    
    print("🧪 Testing all endpoints...")
    print("=" * 50)
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health error: {e}")
    
    # Test data status
    try:
        response = requests.get(f"{base_url}/api/data-status")
        print(f"Data Status: {response.status_code}")
        if response.status_code == 200:
            print(f"  {response.json()}")
        else:
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"Data Status error: {e}")
    
    # Test players endpoint
    try:
        response = requests.get(f"{base_url}/api/players")
        print(f"Players: {response.status_code}")
        if response.status_code == 200:
            players = response.json().get('players', [])
            print(f"  Found {len(players)} players")
            if players:
                print(f"  Sample: {players[0]}")
        else:
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"Players error: {e}")
    
    # Test rankings
    try:
        response = requests.post(f"{base_url}/api/rankings", 
                               json={"position": "QB", "week": 8})
        print(f"Rankings: {response.status_code}")
        if response.status_code == 200:
            rankings = response.json().get('rankings', [])
            print(f"  Found {len(rankings)} rankings")
            if rankings:
                print(f"  Sample: {rankings[0]}")
        else:
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"Rankings error: {e}")

if __name__ == "__main__":
    test_all_endpoints()