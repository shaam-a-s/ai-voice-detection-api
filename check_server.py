#!/usr/bin/env python3
"""
Diagnostic script to check Render server status.
"""
import requests
import time

base_url = "https://ai-voice-detection-api-ql9u.onrender.com"

print("🔍 Checking Render server status...\n")

# Test 1: Root endpoint
print("1. Testing root endpoint (/)...")
try:
    response = requests.get(f"{base_url}/", timeout=10)
    print(f"   ✅ Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print()

# Test 2: Health check
print("2. Testing health endpoint (/healthz)...")
try:
    response = requests.get(f"{base_url}/healthz", timeout=120)
    print(f"   ✅ Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {response.json()}")
        if response.json().get('model_loaded'):
            print("   ✅ Model is loaded and ready!")
        else:
            print("   ⚠️ Model not loaded!")
except requests.exceptions.Timeout:
    print("   ⏱️ Timeout - server is waking up (this is normal)")
    print("   Wait 30 seconds and try again...")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print()

# Test 3: Check if server is deploying
print("3. Recommendations:")
print("   - Go to: https://dashboard.render.com/")
print("   - Check if deployment is 'Live' (green)")
print("   - If 'Deploying', wait 2-3 minutes")
print("   - Check 'Logs' tab for errors")
