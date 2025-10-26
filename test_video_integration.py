#!/usr/bin/env python3
"""
Test script to verify video matching integration in FastAPI
"""
import requests
import json
import time

def test_video_integration():
    """Test the video matching integration"""
    test_cases = [
        {
            "question": "What are the symptoms of jaundice in newborns?",
            "expected_video": True,
            "description": "Should find video about jaundice/risk factors"
        },
        {
            "question": "How to measure baby's temperature?",
            "expected_video": True,
            "description": "Should find video about temperature measurement"
        },
        {
            "question": "What is the treatment for malaria?",
            "expected_video": False,
            "description": "Should not find video (not in video topics)"
        },
        {
            "question": "Hi there, how are you?",
            "expected_video": False,
            "description": "Chitchat - should not find video"
        }
    ]
    
    print("="*80)
    print("VIDEO MATCHING INTEGRATION TEST")
    print("="*80)
    print("Testing video URL integration in FastAPI responses")
    print("="*80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {test_case['question']}")
        print(f"Description: {test_case['description']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        url = "http://localhost:8000/chat"
        params = {
            "user_id": f"video_test_{i}",
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
                video_url = data.get('video_url', 'NOT_FOUND')
                
                print(f"Intent: {intent}")
                print(f"Answer Length: {answer_length}")
                print(f"Video URL: {video_url}")
                
                # Check if video URL is present
                if video_url and video_url != 'NOT_FOUND' and video_url is not None:
                    print("FOUND VIDEO: Video URL present in response")
                    if test_case['expected_video']:
                        print("SUCCESS: Expected video found")
                    else:
                        print("UNEXPECTED: Video found when not expected")
                else:
                    print("NO VIDEO: No video URL in response")
                    if not test_case['expected_video']:
                        print("SUCCESS: No video as expected")
                    else:
                        print("FAILED: Expected video but none found")
            else:
                print(f"HTTP ERROR: {response.status_code}")
                print(f"Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"REQUEST ERROR: {e}")
    
    print("\n" + "="*80)
    print("VIDEO INTEGRATION TEST SUMMARY")
    print("="*80)
    print("EXPECTED API RESPONSE FORMAT:")
    print('{"answer": "...", "intent": "answer", "follow_up": "...", "video_url": "..."}')
    print("="*80)
    print("VIDEO URL FIELD:")
    print("- Contains video URL if relevant video found")
    print("- Contains null if no relevant video found")
    print("- Only added for medical answers (not chitchat)")
    print("="*80)
    print("PERFORMANCE EXPECTATIONS:")
    print("- Video matching: ~1-3s additional time")
    print("- Total response time: ~6-10s (including video matching)")
    print("="*80)

if __name__ == "__main__":
    test_video_integration()
