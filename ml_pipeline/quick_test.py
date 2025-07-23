#!/usr/bin/env python3
"""
Quick test to verify the monitoring fixes
"""

import requests
import json

def test_inference():
    """Test the inference endpoint"""
    url = "http://localhost:8000/infer"
    
    payload = {
        "prompt": "What is the capital of France?",
        "max_new_tokens": 10,
        "adapter_version": "v1.0",
        "base_model": "microsoft/DialoGPT-medium"
    }
    
    params = {"api_key": "test-key"}
    
    try:
        response = requests.post(url, json=payload, params=params)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Result: {data.get('result')}")
            print(f"📊 Metadata: {data.get('metadata')}")
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health():
    """Test health endpoint"""
    url = "http://localhost:8000/health"
    
    try:
        response = requests.get(url)
        print(f"Health Status: {response.status_code}")
        print(f"Health Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Quick Monitoring Test")
    print("=" * 30)
    
    # Test health first
    print("\n🏥 Testing health...")
    health_ok = test_health()
    
    if health_ok:
        print("\n🧠 Testing inference...")
        inference_ok = test_inference()
        
        if inference_ok:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Inference test failed!")
    else:
        print("\n❌ Health test failed!") 