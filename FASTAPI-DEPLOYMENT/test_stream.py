#!/usr/bin/env python3
"""
Test script for SSE (Server-Sent Events) streaming endpoint
This is the WORKING streaming option since WebSocket has 404 issues
"""

import httpx
import json
import sys

BASE_URL = "https://myfastapihub.duckdns.org/deploy"

def test_sse_stream(user_id: str = "test_user", message: str = "What is jaundice?"):
    """
    Test the SSE streaming endpoint
    """
    url = f"{BASE_URL}/chat-stream"
    params = {
        "user_id": user_id,
        "message": message
    }
    
    print("=" * 80)
    print("SSE Streaming Test")
    print("=" * 80)
    print(f"URL: {url}")
    print(f"User ID: {user_id}")
    print(f"Message: {message}")
    print("-" * 80)
    print("📥 Streaming response...\n")
    
    try:
        with httpx.stream("GET", url, params=params, timeout=60.0) as response:
            if response.status_code != 200:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                return False
            
            token_count = 0
            full_response = ""
            metadata_received = False
            
            for line in response.iter_lines():
                if not line.strip():
                    continue
                
                # SSE format: "data: {...}\n\n"
                if line.startswith("data: "):
                    data_str = line[6:].strip()  # Remove "data: " prefix
                    
                    if data_str == "[DONE]":
                        print("\n\n✅ Stream complete!")
                        break
                    
                    try:
                        data = json.loads(data_str)
                        msg_type = data.get("type", "unknown")
                        
                        if msg_type == "token":
                            content = data.get("content", "")
                            print(content, end="", flush=True)
                            full_response += content
                            token_count += 1
                        
                        elif msg_type == "metadata":
                            metadata_received = True
                            print("\n\n" + "=" * 80)
                            print("📊 METADATA RECEIVED:")
                            print("=" * 80)
                            print(json.dumps(data, indent=2))
                            print("=" * 80)
                        
                        elif msg_type == "error":
                            print(f"\n\n❌ ERROR: {data.get('message', 'Unknown error')}")
                        
                        else:
                            print(f"\n⚠️  Unknown type: {msg_type}")
                    
                    except json.JSONDecodeError as e:
                        print(f"\n⚠️  Failed to parse JSON: {e}")
                        print(f"Raw: {data_str[:200]}")
            
            print("\n" + "=" * 80)
            print("📝 SUMMARY:")
            print("=" * 80)
            print(f"✅ Status: HTTP {response.status_code}")
            print(f"✅ Tokens received: {token_count}")
            print(f"✅ Metadata received: {'Yes' if metadata_received else 'No'}")
            print(f"✅ Response length: {len(full_response)} characters")
            print("=" * 80)
            return True
    
    except httpx.RequestError as e:
        print(f"❌ Request error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    user_id = sys.argv[1] if len(sys.argv) > 1 else "test_user"
    message = sys.argv[2] if len(sys.argv) > 2 else "What is jaundice?"
    
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("Usage: python test_stream.py [user_id] [message]")
        print("\nExample:")
        print("  python test_stream.py test123 'What causes fever?'")
        sys.exit(0)
    
    success = test_sse_stream(user_id, message)
    sys.exit(0 if success else 1)

