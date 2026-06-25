import requests

API_URL = "http://localhost:8000/api/v1"

def test_api():
    query = "Where is SEAT located?"
    print(f"Testing Query: '{query}'")
    
    for version in ["v0", "v2", "v3", "v4", "v5"]:
        print(f"\n--- Testing version: {version} ---")
        try:
            payload = {
                "question": query,
                "version": version,
                "escalation_threshold": 0.70
            }
            res = requests.post(f"{API_URL}/ask", json=payload)
            print(f"Status Code: {res.status_code}")
            if res.status_code == 200:
                data = res.json()
                print(f"Decision: {data.get('decision')}")
                print(f"Confidence Score: {data.get('confidence_score')}")
                print(f"Answer: {data.get('answer')}")
                print(f"Raw Answer: {data.get('raw_answer')}")
                print(f"Sources: {data.get('sources')}")
            else:
                print(f"Error: {res.text}")
        except Exception as e:
            print(f"Request failed: {str(e)}")

if __name__ == "__main__":
    test_api()
