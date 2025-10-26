#!/usr/bin/env python3
"""
Test the enhanced safe_json_parse function
"""
import json
import re

def safe_json_parse(text):
    """Enhanced JSON parsing with multiple fallback strategies"""
    if not text or not isinstance(text, str):
        return None
    
    # Strategy 1: Try to find JSON between first { and last }
    try:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx:end_idx+1]
            obj = json.loads(json_str)
            return obj
    except Exception as e:
        print(f"[safe_json_parse] Strategy 1 failed: {e}")
    
    # Strategy 2: Try to find JSON array between first [ and last ]
    try:
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx:end_idx+1]
            obj = json.loads(json_str)
            return obj
    except Exception as e:
        print(f"[safe_json_parse] Strategy 2 failed: {e}")
    
    # Strategy 3: Try to extract JSON from markdown code blocks
    try:
        # Look for ```json ... ``` or ``` ... ```
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            obj = json.loads(json_match.group(1))
            return obj
    except Exception as e:
        print(f"[safe_json_parse] Strategy 3 failed: {e}")
    
    # Strategy 4: Try to find any valid JSON in the text
    try:
        # Find all potential JSON objects
        json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        for match in json_matches:
            try:
                obj = json.loads(match)
                return obj
            except:
                continue
    except Exception as e:
        print(f"[safe_json_parse] Strategy 4 failed: {e}")
    
    # Strategy 5: Try to parse the entire text as JSON
    try:
        obj = json.loads(text.strip())
        return obj
    except Exception as e:
        print(f"[safe_json_parse] Strategy 5 failed: {e}")
    
    print(f"[safe_json_parse] All strategies failed. Raw text: {text[:200]}...")
    return None

def test_json_parsing():
    """Test the enhanced JSON parsing with various Gemini response formats"""
    
    test_cases = [
        {
            "name": "Perfect JSON",
            "input": '{"judgments":[{"index":0,"topic_match":"strong","sufficient":true}]}',
            "expected": True
        },
        {
            "name": "JSON with extra text",
            "input": 'Here is the analysis: {"judgments":[{"index":0,"topic_match":"strong","sufficient":true}]} End of analysis.',
            "expected": True
        },
        {
            "name": "JSON in markdown",
            "input": '```json\n{"judgments":[{"index":0,"topic_match":"strong","sufficient":true}]}\n```',
            "expected": True
        },
        {
            "name": "Multiple JSON objects",
            "input": 'First: {"invalid":"data"} Second: {"judgments":[{"index":0,"topic_match":"strong","sufficient":true}]}',
            "expected": True
        },
        {
            "name": "Invalid JSON",
            "input": 'This is not JSON at all',
            "expected": False
        },
        {
            "name": "Empty string",
            "input": '',
            "expected": False
        },
        {
            "name": "None input",
            "input": None,
            "expected": False
        }
    ]
    
    print("="*80)
    print("JSON PARSING TEST")
    print("="*80)
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        
        result = safe_json_parse(test_case['input'])
        success = (result is not None) == test_case['expected']
        
        if success:
            print("PASSED")
            passed += 1
        else:
            print("FAILED")
            print(f"Expected: {test_case['expected']}, Got: {result is not None}")
        
        if result:
            print(f"Parsed result: {result}")
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("ALL JSON PARSING TESTS PASSED!")
    else:
        print("Some JSON parsing tests failed!")
    
    print("="*80)

if __name__ == "__main__":
    test_json_parsing()
