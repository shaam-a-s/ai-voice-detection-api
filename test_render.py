#!/usr/bin/env python3
"""
Quick test script to verify the updated API works locally before pushing to GUVI.
"""
import requests
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "b64.txt")

with open(file_path, "r", encoding="utf-8") as f:
    b64_string = f.read().strip()

# Test the deployed Render URL
url = "https://ai-voice-detection-api-ql9u.onrender.com/api/voice-detection"

payload = {
    "language": "en",
    "audioFormat": "mp3",
    "audioBase64": b64_string
}

headers = {
    "x-api-key": "sk_test_123456789",
    "Content-Type": "application/json"
}

print("🧪 Testing Render deployment with improved logging...")
print(f"URL: {url}")
print(f"Audio base64 length: {len(b64_string)} characters\n")

try:
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    print(f"✅ Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ SUCCESS! Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ ERROR Response:")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)
            
except requests.exceptions.Timeout:
    print("⏱️ REQUEST TIMED OUT - Server might be sleeping (cold start)")
    print("Run wake_server.py first, then try again.")
except Exception as e:
    print(f"❌ Request failed: {e}")
