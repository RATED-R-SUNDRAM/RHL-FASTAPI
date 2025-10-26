#!/usr/bin/env python3
"""
Test script to verify summarization works correctly
"""
import requests
import json

def test_summarization():
    """Test that summarization works when history exceeds 3 pairs"""
    # Test with multiple questions to trigger summarization
    test_questions = [
        "What are the symptoms of jaundice in newborns?",
        "What causes jaundice in babies?", 
        "How to treat newborn jaundice?",
        "What are the danger signs of jaundice?",
        "How long does jaundice last in newborns?"  # This should trigger summarization
    ]
    
    print("="*80)
    print("TESTING SUMMARIZATION (History > 3 pairs)")
    print("="*80)
    
    user_id = "summarization_test"
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- QUESTION {i}: '{question}' ---")
        
        url = "http://localhost:8000/chat"
        params = {
            "user_id": user_id,
            "message": question
        }
        
        try:
            response = requests.get(url, params=params)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Intent: {data.get('intent', 'No intent')}")
                print(f"Answer: {data.get('answer', 'No answer')[:100]}...")
                
                if i >= 4:  # After 4th question, summarization should trigger
                    print("🔍 CHECK SERVER CONSOLE FOR SUMMARIZATION DEBUG:")
                    print("- Should see: [SUMMARIZATION] History exceeds 3 pairs, summarizing older entries...")
                    print("- Should see: [SUMMARIZATION] Using GEMINI for summarization...")
                    print("- Should see: [SUMMARIZATION] Generated summary: ...")
                    print("- Should see: [SUMMARIZATION] Reduced history to 3 pairs")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Error making request: {e}")
    
    print("\n" + "="*80)
    print("SUMMARIZATION TEST COMPLETE")
    print("="*80)
    print("Expected behavior:")
    print("1. First 3 questions: Normal processing")
    print("2. 4th+ questions: Should trigger summarization of older history")
    print("3. History should be reduced to last 3 pairs")
    print("4. Summary should be generated using Gemini")
    print("="*80)

if __name__ == "__main__":
    test_summarization()
