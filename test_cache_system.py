#!/usr/bin/env python3
"""
Test script to verify cache system works correctly
"""
import requests
import json

def test_cache_system():
    """Test cache system with various queries"""
    test_cases = [
        "What are the symptoms of jaundice in newborns?",
        "What causes jaundice in babies?", 
        "How to treat newborn jaundice?",
        "What are the danger signs of jaundice?",
        "How long does jaundice last in newborns?",
        "What is physiological jaundice?",
        "What is pathological jaundice?",
        "How to prevent jaundice in newborns?",
        "What is phototherapy for jaundice?",
        "When to worry about newborn jaundice?"
    ]
    
    print("="*80)
    print("TESTING CACHE SYSTEM")
    print("="*80)
    
    for i, message in enumerate(test_cases, 1):
        print(f"\n--- TEST CASE {i}: '{message}' ---")
        
        url = "http://localhost:8000/chat"
        params = {
            "user_id": f"cache_test_{i}",
            "message": message
        }
        
        try:
            response = requests.get(url, params=params)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Intent: {data.get('intent', 'No intent')}")
                print(f"Answer: {data.get('answer', 'No answer')[:100]}...")
                
                # Check if it was handled as cached
                if data.get('intent') == 'cached':
                    print("✅ CACHE HIT - Answer served from cache")
                else:
                    print("❌ CACHE MISS - Answer served from RAG pipeline")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Error making request: {e}")
    
    print("\n" + "="*80)
    print("CHECK YOUR SERVER CONSOLE FOR CACHE DEBUG OUTPUT:")
    print("You should see:")
    print("- [CACHE] Loading cache from Excel file: D:\\RHL-WH\\RHL-FASTAPI\\FILES\\cache_questions.xlsx...")
    print("- [CACHE] Computing embeddings for cached questions...")
    print("- [CACHE_CHECK] Step 1: Encoding query...")
    print("- [CACHE_CHECK] Step 2: Computing similarities...")
    print("- [CACHE_CHECK] Step 3: Finding best match...")
    print("- [CACHE_CHECK] ✅ CACHE HIT! or ❌ CACHE MISS!")
    print("- [PIPELINE] ✅ CACHE HIT! Returning cached answer")
    print("="*80)

if __name__ == "__main__":
    test_cache_system()
