import requests
import time

# Wake up the server before GUVI tests it
url = "https://ai-voice-detection-api-ql9u.onrender.com/healthz"

print("Waking up Render server...")
print("This may take up to 2 minutes on first request...")

try:
    response = requests.get(url, timeout=120)
    if response.status_code == 200:
        print("✅ Server is AWAKE and ready!")
        print("You can now submit to GUVI immediately.")
    else:
        print(f"⚠️ Unexpected status: {response.status_code}")
except requests.exceptions.Timeout:
    print("❌ Server took too long. Try again in 30 seconds.")
except Exception as e:
    print(f"Error: {e}")
