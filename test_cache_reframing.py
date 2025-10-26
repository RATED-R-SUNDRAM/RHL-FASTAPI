#!/usr/bin/env python3
"""
Test script to verify cache system with enhanced debug output and answer reframing
"""

import requests
import json
import time

def test_cache_reframing():
    """Test the cache system with reframing functionality"""
    
    base_url = "http://localhost:8000"
    
    # Test cases to verify reframing works
    test_cases = [
        {
            "question": "What are symptoms of jaundice?",
            "expected_cache_hit": True,
            "description": "Direct match - should return cached answer as-is"
        },
        {
            "question": "What causes jaundice?", 
            "expected_cache_hit": True,
            "description": "Semantic match but different focus - should reframe cached answer to focus on causes"
        },
        {
            "question": "How to care for newborn eyes?",
            "expected_cache_hit": True,
            "description": "Direct match - should return cached answer as-is"
        },
        {
            "question": "What are signs of dehydration in babies?",
            "expected_cache_hit": True,
            "description": "Semantic match - should reframe cached answer to focus on signs"
        },
        {
            "question": "How to measure baby temperature?",
            "expected_cache_hit": True,
            "description": "Direct match - should return cached answer as-is"
        },
        {
            "question": "What causes fever in infants?",
            "expected_cache_hit": True,
            "description": "Semantic match - should reframe cached answer to focus on causes"
        }
    ]
    
    print("="*80)
    print("CACHE SYSTEM REFRAMING TEST")
    print("="*80)
    print("Testing cache hits with answer reframing...")
    print("Look for detailed debug output in server console!")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"--- TEST CASE {i}: {test_case['description']} ---")
        print(f"Question: {test_case['question']}")
        
        try:
            # Make API request
            start_time = time.time()
            response = requests.get(
                f"{base_url}/chat",
                params={
                    "user_id": f"reframe_test_user_{i}",
                    "message": test_case['question']
                },
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                response_time = end_time - start_time
                
                print(f"Response time: {response_time:.2f} seconds")
                print(f"Answer length: {len(data.get('answer', ''))}")
                print(f"Video URL: {data.get('video_url', 'None')}")
                
                # Determine if it was a cache hit based on response time
                if response_time < 3.0:
                    print("RESULT: CACHE HIT (fast response)")
                else:
                    print("RESULT: CACHE MISS (slow response - went through RAG)")
                
                # Print answer for analysis
                answer = data.get('answer', '')
                print(f"Answer preview: {answer[:200]}...")
                
                # Check if answer seems reframed (this is subjective)
                if "causes" in test_case['question'].lower() and "causes" in answer.lower():
                    print("REFRAMING: Answer appears to focus on causes ✓")
                elif "signs" in test_case['question'].lower() and ("signs" in answer.lower() or "symptoms" in answer.lower()):
                    print("REFRAMING: Answer appears to focus on signs ✓")
                elif "how to" in test_case['question'].lower() and ("how" in answer.lower() or "to" in answer.lower()):
                    print("REFRAMING: Answer appears to focus on procedures ✓")
                else:
                    print("REFRAMING: Answer structure unclear")
                
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        print("-" * 60)
        print()
    
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("Check server console for detailed cache debug output!")
    print("Look for:")
    print("- CACHE SYSTEM BLOCK")
    print("- BERT similarity computation timing")
    print("- LLM verification results")
    print("- Answer reframing timing")
    print("- Reframed answer previews")
    print("="*80)

if __name__ == "__main__":
    test_cache_reframing()

