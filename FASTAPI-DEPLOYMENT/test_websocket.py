#!/usr/bin/env python3
"""
Test Script for RHL FastAPI Chat Endpoints
- Tests WebSocket endpoint (/chat-ws)
- Tests SSE streaming endpoint (/chat-stream) - SHOWS MULTIPLE JSON RESPONSES

Requirements:
    pip install websockets httpx

Usage:
    # Test SSE streaming (shows multiple JSON responses in real-time):
    python test_websocket.py --test-stream "What is jaundice?"
    # For best real-time visualization, use unbuffered mode:
    python -u test_websocket.py --test-stream "BP"
    
    # Test WebSocket:
    python test_websocket.py "What is jaundice?"
    
    # Test all URLs:
    python test_websocket.py --test-all
"""

import asyncio
import websockets
import json
import sys
from datetime import datetime

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("⚠️  Warning: httpx not installed. HTTP endpoint testing will be skipped.")
    print("   Install with: pip install httpx")

# WebSocket URL - Update this based on your deployment
WEBSOCKET_URL = "wss://myfastapihub.duckdns.org/deploy/chat-ws"

# Alternative URLs to try if the first one fails:
ALTERNATIVE_URLS = [
    "wss://myfastapihub.duckdns.org/chat-ws",  # If app is at root
    "ws://myfastapihub.duckdns.org/deploy/chat-ws",  # If HTTP (not HTTPS)
]

async def test_websocket(url: str, user_id: str = "test_user", message: str = "What is jaundice?"):
    """
    Test WebSocket connection and message streaming
    
    Args:
        url: WebSocket URL to connect to
        user_id: Test user ID
        message: Test message to send
    """
    print("=" * 80)
    print(f"WebSocket Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"URL: {url}")
    print(f"User ID: {user_id}")
    print(f"Message: {message}")
    print("-" * 80)
    
    try:
        print("🔄 Connecting to WebSocket...")
        async with websockets.connect(url, ping_interval=None) as websocket:
            print("✅ Connected successfully!")
            print("-" * 80)
            
            # Send message
            payload = {
                "user_id": user_id,
                "message": message
            }
            print(f"📤 Sending: {json.dumps(payload, indent=2)}")
            await websocket.send(json.dumps(payload))
            print("-" * 80)
            print("📥 Receiving stream...\n")
            
            # Track response
            token_count = 0
            full_response = ""
            metadata_received = False
            
            # Receive messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    
                    if msg_type == "token":
                        # Stream tokens in real-time
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
                    
                    elif msg_type == "done":
                        print("\n\n✅ Stream complete!")
                        print(f"Total tokens received: {token_count}")
                        break
                    
                    else:
                        print(f"\n⚠️  Unknown message type: {msg_type}")
                        print(f"Data: {data}")
                
                except json.JSONDecodeError as e:
                    print(f"\n⚠️  Failed to parse JSON: {e}")
                    print(f"Raw message: {message[:200]}")
            
            print("\n" + "=" * 80)
            print("📝 SUMMARY:")
            print("=" * 80)
            print(f"✅ Connection: Successful")
            print(f"✅ Tokens received: {token_count}")
            print(f"✅ Metadata received: {'Yes' if metadata_received else 'No'}")
            print(f"✅ Full response length: {len(full_response)} characters")
            print("=" * 80)
            return True
            
    except websockets.exceptions.InvalidURI as e:
        print(f"❌ Invalid WebSocket URL: {e}")
        return False
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ Connection closed: {e}")
        return False
    except (websockets.exceptions.InvalidStatusCode, websockets.exceptions.InvalidStatus) as e:
        status_code = getattr(e, 'status_code', None) or (str(e).split('HTTP ')[1].split()[0] if 'HTTP' in str(e) else 'Unknown')
        print(f"❌ Invalid status code: {e}")
        print(f"   HTTP Status: {status_code}")
        print("   This might mean:")
        print("     - The endpoint doesn't exist (404)")
        print("     - The path is incorrect")
        print("     - WebSocket upgrades are not configured in reverse proxy")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_http_endpoints():
    """Test HTTP endpoints to understand the routing structure"""
    if not HTTPX_AVAILABLE:
        print("⚠️  Skipping HTTP endpoint test (httpx not available)")
        return
    
    print("\n🔍 Testing HTTP endpoints to understand routing...\n")
    
    base_url = "https://myfastapihub.duckdns.org"
    test_paths = [
        "/deploy",
        "/deploy/",
        "/deploy/docs",
        "/deploy/openapi.json",
        "/deploy/chat",
        "/deploy/chat-ws",
        "/",
        "/docs",
        "/chat",
        "/chat-ws",
    ]
    
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        for path in test_paths:
            url = f"{base_url}{path}"
            try:
                response = await client.get(url)
                status_emoji = "✅" if response.status_code == 200 else "⚠️" if response.status_code < 400 else "❌"
                print(f"{status_emoji} {url} -> HTTP {response.status_code}")
                if response.status_code == 200 and path.endswith("docs"):
                    print(f"   📄 Found FastAPI docs! Base path might be: {path.rsplit('/docs', 1)[0]}")
            except Exception as e:
                print(f"❌ {url} -> Error: {e}")
    
    print()

async def test_http_chat_endpoint():
    """Test if the regular HTTP /chat endpoint works"""
    if not HTTPX_AVAILABLE:
        return False
    
    print("\n🔍 Testing HTTP /chat endpoint to confirm routing...\n")
    
    test_urls = [
        "https://myfastapihub.duckdns.org/deploy/chat?user_id=test&message=test",
        "https://myfastapihub.duckdns.org/chat?user_id=test&message=test",
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for url in test_urls:
            try:
                print(f"Testing: {url}")
                response = await client.get(url)
                if response.status_code == 200:
                    print(f"✅ HTTP /chat endpoint works! Status: {response.status_code}")
                    print(f"   This confirms the base path. WebSocket should be at same base path + /chat-ws")
                    return True
                else:
                    print(f"⚠️  HTTP /chat returned: {response.status_code}")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print("⚠️  HTTP /chat endpoint not accessible. This might indicate routing issues.\n")
    return False

async def test_multiple_urls():
    """Try multiple WebSocket URLs to find the correct one"""
    print("\n🔍 Testing multiple WebSocket URLs...\n")
    
    # First check if HTTP endpoint works to understand routing
    http_works = await test_http_chat_endpoint()
    
    test_message = "What is jaundice?"
    all_urls = [WEBSOCKET_URL] + ALTERNATIVE_URLS
    
    for url in all_urls:
        print(f"\n{'='*80}")
        print(f"Testing: {url}")
        print('='*80)
        
        success = await test_websocket(url, message=test_message)
        
        if success:
            print(f"\n✅ SUCCESS! Working URL: {url}")
            return url
        else:
            print(f"\n❌ Failed: {url}")
            print("\nTrying next URL...\n")
    
    print("\n" + "=" * 80)
    print("❌ DIAGNOSIS: All WebSocket URLs failed with HTTP 404")
    print("=" * 80)
    print("\n🔧 TROUBLESHOOTING STEPS:")
    print("\n1. Check if FastAPI app is mounted at /deploy:")
    print("   - If using a reverse proxy (nginx/traefik), check the mount path")
    print("   - The HTTP endpoint should work: https://myfastapihub.duckdns.org/deploy/chat")
    print("\n2. Verify WebSocket endpoint exists in code:")
    print("   - Check that @app.websocket('/chat-ws') is defined")
    print("   - The endpoint should be accessible at: /deploy/chat-ws (if mounted at /deploy)")
    print("\n3. Check reverse proxy WebSocket configuration:")
    print("   - Nginx: Ensure 'proxy_set_header Upgrade $http_upgrade;' is set")
    print("   - Traefik: WebSocket should work automatically")
    print("   - Check proxy timeout settings for long connections")
    print("\n4. Test locally first:")
    print("   - Run FastAPI locally: uvicorn rhl_fastapi_deploy:app --reload")
    print("   - Test: ws://localhost:8000/chat-ws")
    print("\n5. Check deployment logs:")
    print("   - Look for WebSocket connection attempts in server logs")
    print("   - Check if 404 errors appear when connecting")
    print("=" * 80)
    return None

async def test_sse_stream(user_id: str = "test_user", message: str = "What is jaundice?"):
    """
    Test the SSE streaming endpoint and show multiple JSON responses in real-time
    This demonstrates that tokens arrive as the model generates them
    """
    if not HTTPX_AVAILABLE:
        print("❌ httpx not available. Install with: pip install httpx")
        return False
    
    url = "https://myfastapihub.duckdns.org/deploy/chat-stream"
    params = {
        "user_id": user_id,
        "message": message
    }
    
    print("=" * 80)
    print("SSE STREAMING TEST - Real-Time Token Visualization")
    print("=" * 80)
    print(f"URL: {url}")
    print(f"User ID: {user_id}")
    print(f"Message: {message}")
    print("-" * 80)
    print("📥 Starting stream... Tokens will appear as they arrive (real-time)\n")
    print("=" * 80)
    sys.stdout.flush()
    
    try:
        # Disable buffering for real-time output
        import time
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("GET", url, params=params) as response:
                if response.status_code != 200:
                    print(f"❌ Error: HTTP {response.status_code}")
                    print(f"Response: {await response.aread()}")
                    sys.stdout.flush()
                    return False
                
                token_count = 0
                json_count = 0
                full_response = ""
                metadata_received = False
                start_time = time.time()
                first_token_time = None
                
                print("\n🔄 STREAMING MODE - Watch tokens appear in real-time:\n")
                print("=" * 80)
                print("📝 ACCUMULATED TEXT (growing as tokens arrive):")
                print("=" * 80)
                print("", end="", flush=True)  # Start on new line
                sys.stdout.flush()
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    # SSE format: "data: {...}\n\n"
                    if line.startswith("data: "):
                        data_str = line[6:].strip()  # Remove "data: " prefix
                        json_count += 1
                        
                        if data_str == "[DONE]":
                            print("\n\n" + "=" * 80)
                            print("✅ Stream complete! Received [DONE] signal")
                            sys.stdout.flush()
                            break
                        
                        try:
                            data = json.loads(data_str)
                            msg_type = data.get("type", "unknown")
                            
                            if msg_type == "token":
                                content = data.get("content", "")
                                
                                # Record first token time
                                if first_token_time is None:
                                    first_token_time = time.time()
                                    elapsed = first_token_time - start_time
                                    print(f"\n⚡ First token arrived after {elapsed:.2f}s\n")
                                    sys.stdout.flush()
                                
                                # Print token content immediately (typing effect)
                                print(content, end="", flush=True)
                                sys.stdout.flush()
                                
                                full_response += content
                                token_count += 1
                                
                                # Show JSON details every 20 tokens (less verbose)
                                if token_count % 20 == 0:
                                    print(f"\n   [JSON #{json_count}: token, content='{content[:30]}...']", flush=True)
                                    sys.stdout.flush()
                            
                            elif msg_type == "metadata":
                                metadata_received = True
                                print("\n\n" + "=" * 80)
                                print("📊 FINAL METADATA RECEIVED:")
                                print("=" * 80)
                                print(json.dumps(data, indent=2))
                                print("=" * 80)
                                sys.stdout.flush()
                            
                            elif msg_type == "error":
                                print(f"\n\n❌ ERROR: {data.get('message', 'Unknown error')}")
                                sys.stdout.flush()
                        
                        except json.JSONDecodeError as e:
                            print(f"\n⚠️  Failed to parse JSON #{json_count}: {e}")
                            print(f"Raw data: {data_str[:200]}")
                            sys.stdout.flush()
                
                end_time = time.time()
                total_time = end_time - start_time
                
                print("\n\n" + "=" * 80)
                print("📝 STREAMING SUMMARY:")
                print("=" * 80)
                print(f"✅ Total JSON responses received: {json_count}")
                print(f"✅ Token JSON responses: {token_count}")
                print(f"✅ Metadata JSON responses: {'1' if metadata_received else '0'}")
                print(f"✅ Full response length: {len(full_response)} characters")
                if first_token_time:
                    print(f"✅ Time to first token: {first_token_time - start_time:.2f}s")
                print(f"✅ Total streaming time: {total_time:.2f}s")
                print(f"✅ Average tokens per second: {token_count / total_time:.1f}" if total_time > 0 else "")
                print("=" * 80)
                sys.stdout.flush()
                return True
        
    except httpx.RequestError as e:
        print(f"❌ Request error: {e}")
        sys.stdout.flush()
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return False

def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("RHL FastAPI Test Script")
    print("=" * 80)
    
    # Check for SSE stream test
    if "--test-stream" in sys.argv or "--stream" in sys.argv or "-s" in sys.argv:
        # Remove the flag from args
        args = [arg for arg in sys.argv[1:] if arg not in ["--test-stream", "--stream", "-s"]]
        user_id = args[0] if len(args) > 0 else "test_user"
        message = " ".join(args[1:]) if len(args) > 1 else "What is jaundice?"
        result = asyncio.run(test_sse_stream(user_id, message))
        sys.exit(0 if result else 1)
    
    # Check if custom message provided
    test_message = "What is jaundice?"
    if len(sys.argv) > 1:
        test_message = " ".join(sys.argv[1:])
        print(f"Using custom message: {test_message}")
    
    # Check if we should test all URLs
    test_all = "--test-all" in sys.argv or "-a" in sys.argv
    
    if test_all:
        # First test HTTP endpoints to understand routing
        asyncio.run(test_http_endpoints())
        result = asyncio.run(test_multiple_urls())
    else:
        result = asyncio.run(test_websocket(WEBSOCKET_URL, message=test_message))
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

