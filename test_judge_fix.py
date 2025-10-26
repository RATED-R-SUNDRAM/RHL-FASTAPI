#!/usr/bin/env python3
"""
Test script to verify judge_sufficiency fix
"""
import requests
import json
import time

def test_judge_sufficiency_fix():
    """Test that judge_sufficiency is working properly"""
    test_cases = [
        "What are the symptoms of jaundice in newborns?",  # Should trigger judge_sufficiency
        "What causes malaria?",  # Should trigger judge_sufficiency
        "How to treat depression?",  # Should trigger judge_sufficiency
    ]
    
    print("="*80)
    print("JUDGE_SUFFICIENCY FIX TEST")
    print("="*80)
    print("Testing that judge_sufficiency works with Gemini LLM")
    print("="*80)
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {question}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        url = "http://localhost:8000/chat"
        params = {
            "user_id": f"judge_test_{i}",
            "message": question
        }
        
        try:
            response = requests.get(url, params=params)
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"Status Code: {response.status_code}")
            print(f"TOTAL CLIENT TIME: {total_time:.3f}s")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Intent: {data.get('intent', 'No intent')}")
                print(f"Answer Length: {len(data.get('answer', ''))}")
                
                # Check if we got a proper answer
                if data.get('intent') == 'answer' and len(data.get('answer', '')) > 100:
                    print("SUCCESS: Got proper answer from RAG pipeline")
                else:
                    print("WARNING: May not have gone through full RAG pipeline")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Error making request: {e}")
    
    print("\n" + "="*80)
    print("JUDGE_SUFFICIENCY FIX SUMMARY")
    print("="*80)
    print("FIXES APPLIED:")
    print("1. Use gemini_llm directly instead of judge_llm parameter")
    print("2. Simplified judge_prompt for better Gemini compatibility")
    print("3. Added detailed error logging to identify parsing issues")
    print("4. Added fallback mechanism for partial JSON parsing")
    print("5. Enhanced debugging output")
    print("="*80)
    print("EXPECTED SERVER CONSOLE OUTPUT:")
    print("- [judge_sufficiency] Using GEMINI LLM for judging")
    print("- [JUDGE_RAW_RESP]: <Gemini response>")
    print("- [JUDGE_PARSED]: <parsed JSON>")
    print("- [judge_sufficiency] AFTER answer_chunks=X AFTER followup_chunks=Y")
    print("="*80)
    print("ERROR MESSAGES TO WATCH FOR:")
    print("- [judge_sufficiency] ERROR: Failed to parse JSON")
    print("- [judge_sufficiency] ERROR: No 'judgments' key")
    print("- [judge_sufficiency] LLM did not return valid batched judgments")
    print("="*80)

if __name__ == "__main__":
    test_judge_sufficiency_fix()
