import requests
import json

def test_api():
    base_url = 'http://localhost:5000'
    
    # Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f'{base_url}/health')
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {str(e)}")

    # Test claim verification
    print("\n2. Testing claim verification...")
    test_claim = "The Earth is flat"
    try:
        response = requests.post(
            f'{base_url}/api/v1/verify',
            json={"claim": test_claim},
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_api()