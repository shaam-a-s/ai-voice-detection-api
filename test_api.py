import requests
import json

# Read the base64 string we generated in step 1
with open("b64.txt", "r", encoding="utf-8") as f:
    b64_string = f.read().strip()

url = "http://127.0.0.1:8000/api/voice-detection"

payload = {
    "language": "en",
    "audioFormat": "mp3",
    "audioBase64": b64_string
}

headers = {
    "x-api-key": "sk_test_123456789",
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Request failed: {e}")
