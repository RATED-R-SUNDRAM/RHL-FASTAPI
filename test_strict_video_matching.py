#!/usr/bin/env python3
"""
Test script to verify strict video matching
"""
import requests
import json
import time

def test_strict_video_matching():
    """Test the strict video matching system"""
    test_cases = [
        {
            "question": "What are the symptoms of jaundice in newborns?",
            "expected_video": True,
            "description": "Should find video about jaundice/risk factors (high relevance)"
        },
        {
            "question": "How to measure baby's temperature using a thermometer?",
            "expected_video": True,
            "description": "Should find video about temperature measurement (high relevance)"
        },
        {
            "question": "The patient presented with uterine infection, maternal fever, and foul-smelling discharge. Risk factors include prolonged rupture of membranes and chorioamnionitis.",
            "expected_video": True,
            "description": "Should find video about risk factors/infection (high relevance)"
        },
        {
            "question": "Indirect hyperbilirubinemia can result from several factors that lead to decreased bilirubin clearance or increased enterohepatic circulation.",
            "expected_video": False,
            "description": "Should NOT find video (no direct match in video topics)"
        },
        {
            "question": "What is the treatment for malaria?",
            "expected_video": False,
            "description": "Should NOT find video (not in video topics)"
        },
        {
            "question": "Hi there, how are you?",
            "expected_video": False,
            "description": "Chitchat - should NOT find video"
        }
    ]
    
    print("="*80)
    print("STRICT VIDEO MATCHING TEST")
    print("="*80)
    print("Testing strict video matching with 4+ word matches and 80+ LLM score")
    print("="*80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {test_case['question']}")
        print(f"Description: {test_case['description']}")
        print(f"Expected Video: {'YES' if test_case['expected_video'] else 'NO'}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        url = "http://localhost:8000/chat"
        params = {
            "user_id": f"strict_test_{i}",
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
                        print("FAILED: Video found when not expected (too loose matching)")
                else:
                    print("NO VIDEO: No video URL in response")
                    if not test_case['expected_video']:
                        print("SUCCESS: No video as expected (strict matching working)")
                    else:
                        print("FAILED: Expected video but none found (too strict matching)")
            else:
                print(f"HTTP ERROR: {response.status_code}")
                print(f"Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"REQUEST ERROR: {e}")
    
    print("\n" + "="*80)
    print("STRICT MATCHING CRITERIA")
    print("="*80)
    print("1. PRE-FILTERING:")
    print("   - Minimum 4 meaningful word matches (increased from 3)")
    print("   - Stop words removed from matching")
    print("   - Only medical/technical terms count")
    print("")
    print("2. LLM SCORING:")
    print("   - Score 80+ required for video return")
    print("   - 90-100: Perfect match")
    print("   - 80-89: Strong match")
    print("   - Below 80: No video returned")
    print("")
    print("3. RESULT:")
    print("   - Only highly relevant videos returned")
    print("   - No false positives")
    print("   - Better user experience")
    print("="*80)

if __name__ == "__main__":
    test_strict_video_matching()

