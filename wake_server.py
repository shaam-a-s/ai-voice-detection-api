import requests
import time

# Wake up the server before GUVI tests it
url = "https://ai-voice-detection-api-ql9u.onrender.com/healthz"

print("Waking up Render server...")
print("This may take up to 2 minutes on first request...")
print()

max_retries = 3
for attempt in range(1, max_retries + 1):
    try:
        print(f"Attempt {attempt}/{max_retries}...")
        response = requests.get(url, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is AWAKE and ready!")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            print()
            print("🎯 You can now submit to GUVI immediately (within 15 minutes).")
            break
        else:
            print(f"⚠️ Unexpected status: {response.status_code}")
            if attempt < max_retries:
                print("   Retrying in 10 seconds...")
                time.sleep(10)
                
    except requests.exceptions.Timeout:
        print("⏱️ Timeout - Server is starting up...")
        if attempt < max_retries:
            print("   Retrying in 15 seconds...")
            time.sleep(15)
        else:
            print("❌ Server took too long. It might be deploying.")
            print("   Check: https://dashboard.render.com/")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if attempt < max_retries:
            print("   Retrying in 10 seconds...")
            time.sleep(10)
        else:
            print()
            print("Server might be redeploying. Check Render dashboard:")
            print("https://dashboard.render.com/")
