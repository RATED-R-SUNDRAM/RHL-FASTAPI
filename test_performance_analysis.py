#!/usr/bin/env python3
"""
Performance analysis script to identify bottlenecks
"""
import requests
import json
import time

def test_performance_analysis():
    """Test and analyze performance bottlenecks"""
    test_cases = [
        "What are the symptoms of jaundice in newborns?",  # Should be cached
        "What causes jaundice in babies?",  # Should be cached
        "How to treat newborn jaundice?",  # Should be cached
        "What is the treatment for malaria?",  # Should go to RAG
    ]
    
    print("="*80)
    print("PERFORMANCE ANALYSIS - BOTTLENECK IDENTIFICATION")
    print("="*80)
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {question}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        url = "http://localhost:8000/chat"
        params = {
            "user_id": f"perf_test_{i}",
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
                
                # Analyze based on intent
                if data.get('intent') == 'cached':
                    print("CACHE HIT - Should be < 2s")
                    if total_time > 2.0:
                        print("PERFORMANCE ISSUE: Cache hit taking too long!")
                elif data.get('intent') == 'answer':
                    print("RAG PIPELINE - Expected 5-7s")
                    if total_time > 10.0:
                        print("PERFORMANCE ISSUE: RAG taking too long!")
                else:
                    print(f"Other intent: {data.get('intent')}")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Error making request: {e}")
    
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*80)
    print("EXPECTED TIMINGS:")
    print("Cache Hit: < 2s total")
    print("RAG Pipeline: 5-7s total")
    print("Current: 11s (TOO SLOW!)")
    print("="*80)
    print("POTENTIAL BOTTLENECKS TO CHECK:")
    print("1. Check server console for detailed timing breakdown")
    print("2. Embedding model loading (first request)")
    print("3. Pinecone network latency")
    print("4. Gemini API latency")
    print("5. Re-ranking step (FlashRank/CrossEncoder)")
    print("6. Database operations")
    print("7. LLM inference time")
    print("="*80)
    print("DEBUG STEPS:")
    print("1. Check if this is the first request (cold start)")
    print("2. Look for 'embedding.encode took X.XXXs'")
    print("3. Look for 'pinecone.query took X.XXXs'")
    print("4. Look for 'rerank took X.XXXs'")
    print("5. Look for 'synthesis took X.XXXs'")
    print("="*80)

if __name__ == "__main__":
    test_performance_analysis()
