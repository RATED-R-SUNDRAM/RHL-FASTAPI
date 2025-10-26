#!/usr/bin/env python3
"""
Comprehensive test for the combined cache verification and reframing system
"""

import requests
import json
import time

def test_combined_cache_system():
    """Test the combined verification and reframing cache system"""
    
    base_url = "http://localhost:8000"
    
    # Test cases covering various scenarios
    test_cases = [
        {
            "question": "What are symptoms of jaundice?",
            "expected_result": "CACHE_HIT_REFRAMED",
            "description": "Direct match - should return cached answer (may be reframed for better flow)"
        },
        {
            "question": "What causes jaundice?", 
            "expected_result": "CACHE_MISS",
            "description": "Different focus - cached answer about symptoms, query asks for causes"
        },
        {
            "question": "How to care for newborn eyes?",
            "expected_result": "CACHE_HIT_REFRAMED",
            "description": "Direct match - should return cached answer about eye care"
        },
        {
            "question": "What are signs of dehydration in babies?",
            "expected_result": "CACHE_HIT_REFRAMED",
            "description": "Semantic match - should reframe cached answer to focus on signs"
        },
        {
            "question": "How to measure baby temperature?",
            "expected_result": "CACHE_HIT_REFRAMED",
            "description": "Direct match - should return cached answer about temperature measurement"
        },
        {
            "question": "What causes fever in infants?",
            "expected_result": "CACHE_HIT_REFRAMED",
            "description": "Semantic match - should reframe cached answer to focus on causes"
        },
        {
            "question": "How to treat pneumonia?",
            "expected_result": "CACHE_MISS",
            "description": "Not in cache - should trigger RAG pipeline"
        },
        {
            "question": "What are complications of diabetes?",
            "expected_result": "CACHE_MISS",
            "description": "Not in cache - should trigger RAG pipeline"
        },
        {
            "question": "What is the treatment for malaria?",
            "expected_result": "CACHE_MISS",
            "description": "Not in cache - should trigger RAG pipeline"
        }
    ]
    
    print("="*80)
    print("COMBINED CACHE VERIFICATION & REFRAMING TEST")
    print("="*80)
    print("Testing combined LLM verification and reframing...")
    print("Look for detailed debug output in server console!")
    print()
    
    cache_hits = 0
    cache_misses = 0
    total_time = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"--- TEST CASE {i}: {test_case['description']} ---")
        print(f"Question: {test_case['question']}")
        print(f"Expected: {test_case['expected_result']}")
        
        try:
            # Make API request
            start_time = time.time()
            response = requests.get(
                f"{base_url}/chat",
                params={
                    "user_id": f"combined_test_user_{i}",
                    "message": test_case['question']
                },
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                response_time = end_time - start_time
                total_time += response_time
                
                print(f"Response time: {response_time:.2f} seconds")
                print(f"Answer length: {len(data.get('answer', ''))}")
                print(f"Video URL: {data.get('video_url', 'None')}")
                
                # Determine actual result based on response time
                if response_time < 3.0:
                    actual_result = "CACHE_HIT_REFRAMED"
                    cache_hits += 1
                    print("ACTUAL: CACHE HIT (fast response)")
                else:
                    actual_result = "CACHE_MISS"
                    cache_misses += 1
                    print("ACTUAL: CACHE MISS (slow response - went through RAG)")
                
                # Check if result matches expectation
                expected = test_case['expected_result']
                if actual_result == expected:
                    print("RESULT: PASS - Matches expectation")
                else:
                    print(f"RESULT: FAIL - Expected {expected}, got {actual_result}")
                
                # Analyze answer content for reframing quality
                answer = data.get('answer', '')
                print(f"Answer preview: {answer[:200]}...")
                
                # Check for reframing indicators
                if actual_result == "CACHE_HIT_REFRAMED":
                    question_lower = test_case['question'].lower()
                    answer_lower = answer.lower()
                    
                    if "causes" in question_lower and "caused" in answer_lower:
                        print("REFRAMING: Answer appears to focus on causes ✓")
                    elif "signs" in question_lower and ("signs" in answer_lower or "symptoms" in answer_lower):
                        print("REFRAMING: Answer appears to focus on signs ✓")
                    elif "how to" in question_lower and ("how" in answer_lower or "to" in answer_lower):
                        print("REFRAMING: Answer appears to focus on procedures ✓")
                    elif "what are" in question_lower:
                        print("REFRAMING: Answer appears to be descriptive ✓")
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
    
    # Summary
    print("="*80)
    print("COMBINED CACHE SYSTEM TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {len(test_cases)}")
    print(f"Cache hits: {cache_hits}")
    print(f"Cache misses: {cache_misses}")
    print(f"Average response time: {total_time/len(test_cases):.2f} seconds")
    print(f"Cache hit rate: {cache_hits/len(test_cases)*100:.1f}%")
    
    print("\nExpected behavior:")
    print("- Cache hits should be fast (< 3 seconds)")
    print("- Cache misses should be slow (> 5 seconds)")
    print("- Answers should be reframed to match question focus")
    print("- Server console should show detailed cache debug output")
    
    print("\nCheck server console for:")
    print("- CACHE SYSTEM BLOCK")
    print("- BERT similarity computation timing")
    print("- Combined LLM verification and reframing timing")
    print("- LLM response previews")
    print("- NULL responses when cached answer cannot answer query")
    
    print("="*80)

if __name__ == "__main__":
    test_combined_cache_system()

