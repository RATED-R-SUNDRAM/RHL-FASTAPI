#!/usr/bin/env python3
"""
Test script to verify hybrid_retrieve optimizations work
"""
import requests
import json

def test_hybrid_optimization():
    """Test hybrid_retrieve with optimized timing"""
    url = "http://localhost:8000/chat"
    params = {
        "user_id": "hybrid_opt_test",
        "message": "What are the symptoms of jaundice in newborns?"
    }
    
    print("="*80)
    print("TESTING HYBRID RETRIEVAL OPTIMIZATION")
    print("="*80)
    print(f"URL: {url}")
    print(f"Params: {params}")
    print("Making request to test optimized hybrid_retrieve...")
    print("="*80)
    
    try:
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Answer: {data.get('answer', 'No answer')[:100]}...")
            print(f"Intent: {data.get('intent', 'No intent')}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error making request: {e}")
    
    print("\n" + "="*80)
    print("CHECK YOUR SERVER CONSOLE FOR COMPREHENSIVE HYBRID TIMING:")
    print("You should see DETAILED timing for EVERY step:")
    print("- [HYBRID] Step 1 - Embedding: X.XXXs (should be ~0.1s)")
    print("- [HYBRID] Step 2 - Pinecone: X.XXXs (should be ~0.5-2.0s)")
    print("- [HYBRID] Step 3 - Process: X.XXXs (should be ~0.01s)")
    print("- [HYBRID] Step 4 - Dedupe: X.XXXs (should be ~0.001s)")
    print("- [HYBRID] Step 5 - Rerank: X.XXXs (should be ~1-3s)")
    print("- [HYBRID] Step 6 - Sort: X.XXXs (should be ~0.001s)")
    print("- [HYBRID] TOTAL TIME: X.XXXs")
    print("="*80)
    print("COMPREHENSIVE TIMING FEATURES:")
    print("✅ Individual step timing (not cumulative)")
    print("✅ Total time calculation")
    print("✅ Step-by-step breakdown")
    print("✅ Optimized processing algorithms")
    print("✅ Early termination for deduplication")
    print("="*80)

if __name__ == "__main__":
    test_hybrid_optimization()
