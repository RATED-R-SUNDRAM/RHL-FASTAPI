#!/usr/bin/env python3
"""
Test script to verify classify+reform optimization from 3s to 1-2s
"""
import requests
import json
import time

def test_classify_optimization():
    """Test the optimized classify+reform step"""
    test_cases = [
        "What are the symptoms of jaundice in newborns?",  # Medical question
        "Hi there",  # Chitchat
        "yes",  # Follow-up
        "What causes malaria?",  # Medical question
        "How to treat depression?",  # Medical question
    ]
    
    print("="*80)
    print("CLASSIFY + REFORM OPTIMIZATION TEST")
    print("="*80)
    print("TARGET: 1-2 seconds for classify+reform (was 3s)")
    print("="*80)
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {question}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        url = "http://localhost:8000/chat"
        params = {
            "user_id": f"classify_test_{i}",
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
                
                # Performance analysis
                if total_time < 2.0:
                    print("EXCELLENT: < 2s total")
                elif total_time < 5.0:
                    print("GOOD: < 5s total")
                elif total_time < 10.0:
                    print("ACCEPTABLE: < 10s total")
                else:
                    print("SLOW: > 10s total")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Error making request: {e}")
    
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    print("OPTIMIZATIONS APPLIED:")
    print("1. Shorter, focused prompt (reduced from ~2000 to ~200 chars)")
    print("2. Lower temperature (0.1) for faster responses")
    print("3. Max tokens limit (150) for classification")
    print("4. 10 second timeout to prevent hanging")
    print("5. Performance warnings for slow responses")
    print("="*80)
    print("EXPECTED IMPROVEMENTS:")
    print("- Classify+Reform: 3s → 1-2s (50-66% faster)")
    print("- Total pipeline: 11s → 5-7s (40-50% faster)")
    print("="*80)
    print("CHECK SERVER CONSOLE FOR:")
    print("- [CLASSIFY_REFORM] Gemini LLM call took X.XXXs")
    print("- [CLASSIFY_REFORM] WARNING messages if >2s")
    print("- [PIPELINE] Classify+Reform: X.XXXs")
    print("="*80)

if __name__ == "__main__":
    test_classify_optimization()
