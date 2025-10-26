#!/usr/bin/env python3
"""
Test edge cases for the combined cache system
"""

import requests
import json
import time

def test_cache_edge_cases():
    """Test edge cases for the combined cache verification and reframing"""
    
    base_url = "http://localhost:8000"
    
    # Edge case test scenarios
    edge_cases = [
        {
            "question": "What are symptoms of jaundice?",
            "description": "Exact match - should work"
        },
        {
            "question": "What causes jaundice?", 
            "description": "Different focus - cached answer about symptoms, should return NULL"
        },
        {
            "question": "How to treat jaundice?",
            "description": "Different focus - cached answer about symptoms, should return NULL"
        },
        {
            "question": "What are complications of jaundice?",
            "description": "Different focus - cached answer about symptoms, should return NULL"
        },
        {
            "question": "What are signs of jaundice?",
            "description": "Semantic match - symptoms = signs, should reframe"
        },
        {
            "question": "What are jaundice symptoms?",
            "description": "Word order change - should reframe"
        },
        {
            "question": "Tell me about jaundice symptoms",
            "description": "Different phrasing - should reframe"
        }
    ]
    
    print("="*80)
    print("CACHE SYSTEM EDGE CASES TEST")
    print("="*80)
    print("Testing edge cases for combined verification and reframing...")
    print()
    
    for i, test_case in enumerate(edge_cases, 1):
        print(f"--- EDGE CASE {i}: {test_case['description']} ---")
        print(f"Question: {test_case['question']}")
        
        try:
            # Make API request
            start_time = time.time()
            response = requests.get(
                f"{base_url}/chat",
                params={
                    "user_id": f"edge_case_user_{i}",
                    "message": test_case['question']
                },
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                response_time = end_time - start_time
                
                print(f"Response time: {response_time:.2f} seconds")
                
                # Determine if cache hit or miss
                if response_time < 3.0:
                    print("RESULT: CACHE HIT")
                    answer = data.get('answer', '')
                    print(f"Answer preview: {answer[:150]}...")
                    
                    # Check if answer seems appropriate
                    if "jaundice" in answer.lower():
                        print("CONTENT: Answer contains jaundice information ✓")
                    else:
                        print("CONTENT: Answer doesn't contain jaundice information ❌")
                        
                else:
                    print("RESULT: CACHE MISS (went through RAG)")
                    answer = data.get('answer', '')
                    print(f"Answer preview: {answer[:150]}...")
                
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        print("-" * 60)
        print()
    
    print("="*80)
    print("EDGE CASES TEST COMPLETE")
    print("="*80)
    print("Expected behavior:")
    print("- Exact matches: Cache hit with reframed answer")
    print("- Different focus (causes vs symptoms): Cache miss, goes to RAG")
    print("- Semantic matches (symptoms = signs): Cache hit with reframed answer")
    print("- Different phrasings: Cache hit with reframed answer")
    print("="*80)

if __name__ == "__main__":
    test_cache_edge_cases()

