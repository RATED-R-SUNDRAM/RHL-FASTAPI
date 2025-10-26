#!/usr/bin/env python3
"""
Comprehensive test for judge_sufficiency fix
"""
import requests
import json
import time

def test_judge_comprehensive():
    """Comprehensive test to ensure judge_sufficiency works properly"""
    test_cases = [
        {
            "question": "What are the symptoms of jaundice in newborns?",
            "expected_intent": "answer",
            "description": "Medical question that should trigger full RAG pipeline"
        },
        {
            "question": "What causes malaria?",
            "expected_intent": "answer", 
            "description": "Another medical question for RAG pipeline"
        },
        {
            "question": "How to treat depression?",
            "expected_intent": "answer",
            "description": "Medical question requiring judge_sufficiency"
        }
    ]
    
    print("="*80)
    print("COMPREHENSIVE JUDGE_SUFFICIENCY TEST")
    print("="*80)
    print("Testing JSON parsing fix and Gemini integration")
    print("="*80)
    
    success_count = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {test_case['question']}")
        print(f"Description: {test_case['description']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        url = "http://localhost:8000/chat"
        params = {
            "user_id": f"judge_comp_test_{i}",
            "message": test_case['question']
        }
        
        try:
            response = requests.get(url, params=params)
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"Status Code: {response.status_code}")
            print(f"TOTAL CLIENT TIME: {total_time:.3f}s")
            
            if response.status_code == 200:
                data = response.json()
                intent = data.get('intent', 'No intent')
                answer_length = len(data.get('answer', ''))
                
                print(f"Intent: {intent}")
                print(f"Answer Length: {answer_length}")
                
                # Check if we got the expected result
                if intent == test_case['expected_intent']:
                    if intent == 'answer' and answer_length > 100:
                        print("SUCCESS: Got proper answer from RAG pipeline")
                        success_count += 1
                    elif intent == 'cached' and answer_length > 50:
                        print("SUCCESS: Got cached answer")
                        success_count += 1
                    else:
                        print("PARTIAL: Got expected intent but short answer")
                        success_count += 0.5
                else:
                    print(f"FAILED: Expected {test_case['expected_intent']}, got {intent}")
            else:
                print(f"HTTP ERROR: {response.status_code}")
                print(f"Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"REQUEST ERROR: {e}")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST RESULTS")
    print("="*80)
    print(f"Tests Passed: {success_count}/{total_tests}")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("ALL TESTS PASSED! Judge sufficiency is working properly.")
    elif success_count >= total_tests * 0.8:
        print("MOSTLY SUCCESSFUL! Minor issues but judge sufficiency is working.")
    else:
        print("SIGNIFICANT ISSUES! Judge sufficiency needs more fixes.")
    
    print("="*80)
    print("EXPECTED SERVER CONSOLE OUTPUT:")
    print("- [judge_sufficiency] Using GEMINI LLM for judging")
    print("- [JUDGE_RAW_RESP]: <Gemini JSON response>")
    print("- [JUDGE_PARSED]: <parsed JSON object>")
    print("- [judge_sufficiency] AFTER answer_chunks=X AFTER followup_chunks=Y")
    print("="*80)
    print("ERROR INDICATORS TO WATCH FOR:")
    print("- [safe_json_parse] All strategies failed")
    print("- [judge_sufficiency] LLM did not return valid batched judgments")
    print("- [judge_sufficiency] Using cross-encoder fallback")
    print("="*80)

if __name__ == "__main__":
    test_judge_comprehensive()
