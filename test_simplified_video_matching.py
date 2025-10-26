#!/usr/bin/env python3
"""
Test script for the simplified BERT-based video matching system
"""

import requests
import json
import time

def test_video_matching():
    """Test the simplified video matching system"""
    
    base_url = "http://localhost:8000"
    
    # Test cases with expected behavior
    test_cases = [
        {
            "question": "What is eye care for newborns?",
            "expected_keywords": ["eye", "care", "newborn", "medication", "infection"],
            "should_have_video": True,
            "description": "Eye care question - should match eye-related video"
        },
        {
            "question": "How to take baby temperature?",
            "expected_keywords": ["temperature", "baby", "thermometer"],
            "should_have_video": True,
            "description": "Temperature question - should match temperature video"
        },
        {
            "question": "What causes jaundice in babies?",
            "expected_keywords": ["jaundice", "baby", "causes"],
            "should_have_video": False,  # No jaundice-specific video in our dataset
            "description": "Jaundice question - may not have specific video"
        },
        {
            "question": "How to care for umbilical cord?",
            "expected_keywords": ["cord", "umbilical", "care"],
            "should_have_video": True,
            "description": "Cord care question - should match cord care video"
        }
    ]
    
    print("="*80)
    print("TESTING SIMPLIFIED BERT-BASED VIDEO MATCHING SYSTEM")
    print("="*80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- TEST CASE {i}: {test_case['description']} ---")
        print(f"Question: {test_case['question']}")
        
        try:
            # Make API request
            start_time = time.time()
            response = requests.get(
                f"{base_url}/chat",
                params={
                    "user_id": f"test_user_{i}",
                    "message": test_case['question']
                },
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"Response time: {end_time - start_time:.2f} seconds")
                print(f"Answer length: {len(data.get('answer', ''))}")
                print(f"Video URL: {data.get('video_url', 'None')}")
                
                # Check if video URL is present
                has_video = data.get('video_url') is not None
                print(f"Has video: {has_video}")
                
                # Validate expectation
                if test_case['should_have_video']:
                    if has_video:
                        print("✅ PASS: Expected video and got video")
                    else:
                        print("❌ FAIL: Expected video but got None")
                else:
                    if not has_video:
                        print("✅ PASS: Expected no video and got None")
                    else:
                        print("❌ FAIL: Expected no video but got video")
                
                # Print first 200 chars of answer for context
                answer_preview = data.get('answer', '')[:200]
                print(f"Answer preview: {answer_preview}...")
                
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        
        print("-" * 60)
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_video_matching()
