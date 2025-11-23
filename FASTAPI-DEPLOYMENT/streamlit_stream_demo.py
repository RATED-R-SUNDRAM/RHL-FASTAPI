"""
Streamlit demo page to visualize streaming responses from FastAPI.
This demonstrates how the UI would display streaming tokens in real-time.
"""
import streamlit as st
import requests
import json
import time
import os
from typing import Optional

st.set_page_config(page_title="Medical Chatbot - Streaming Demo", layout="wide")

st.title("🏥 Medical Chatbot - Streaming Response Visualization")
st.markdown("This page demonstrates how streaming responses appear in real-time UI")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
# Allow API URL to be set via environment variable for deployment
default_api_url = os.getenv("FASTAPI_URL", "http://localhost:8000")
api_url = st.sidebar.text_input("API Base URL", value=default_api_url)
user_id = st.sidebar.text_input("User ID", value="demo_user")
stream_mode = st.sidebar.radio("Response Mode", ["Streaming", "Standard"], index=0)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'streaming_active' not in st.session_state:
    st.session_state.streaming_active = False

# Display chat history
st.header("💬 Chat Interface")

# Show message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg and msg["metadata"]:
            with st.expander("📎 Response Metadata"):
                st.json(msg["metadata"])
                if msg["metadata"].get('video_url'):
                    st.video(msg["metadata"]['video_url'])
                if msg["metadata"].get('follow_up'):
                    st.info(f"💡 Follow-up suggestion: {msg['metadata']['follow_up']}")

# Chat input
if prompt := st.chat_input("Ask a medical question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Create placeholder for streaming response
    with st.chat_message("assistant"):
        # Timing placeholders
        timing_placeholder = st.empty()
        message_placeholder = st.empty()
        metadata_placeholder = st.empty()
        full_response = ""
        metadata = None  # Initialize metadata
        
        try:
            if stream_mode == "Streaming":
                # Record time when request is sent (frontend perspective)
                request_sent_time = time.time()
                first_token_received = False
                first_token_time = None
                
                # Make streaming request to FastAPI
                stream_url = f"{api_url}/chat-stream"
                params = {
                    "user_id": user_id,
                    "message": prompt
                }
                
                st.session_state.streaming_active = True
                
                # Show initial status
                timing_placeholder.info("⏳ Processing request...")
                
                # Stream the response
                with requests.get(stream_url, params=params, stream=True, timeout=120) as response:
                    if response.status_code != 200:
                        message_placeholder.error(f"Error: {response.status_code} - {response.text}")
                        timing_placeholder.empty()
                        st.session_state.streaming_active = False
                    else:
                        # Process SSE stream
                        metadata = {}
                        is_done = False
                        
                        for line in response.iter_lines():
                            if not st.session_state.streaming_active:
                                break
                            
                            if line:
                                line_str = line.decode('utf-8')
                                
                                # Handle SSE format: "data: <json>\n\n"
                                if line_str.startswith('data: '):
                                    data_str = line_str[6:]  # Remove "data: "
                                    
                                    if data_str == '[DONE]':
                                        is_done = True
                                        break
                                    
                                    try:
                                        data = json.loads(data_str)
                                        
                                        if data.get('type') == 'token':
                                            # Record time when first token arrives
                                            if not first_token_received:
                                                first_token_time = time.time()
                                                time_to_first_token = first_token_time - request_sent_time
                                                first_token_received = True
                                                # Update timing display with clarification
                                                timing_placeholder.success(
                                                    f"⚡ **Time to first token:** `{time_to_first_token:.3f}s` (user experience)\n\n"
                                                    f"💡 *Includes: network latency + backend processing + rendering*"
                                                )
                                            
                                            # Append token to full response
                                            token = data.get('content', '')
                                            full_response += token
                                            
                                            # Update display with typing effect
                                            message_placeholder.markdown(full_response + "▌")
                                        
                                        elif data.get('type') == 'metadata':
                                            # Store metadata for later display
                                            metadata = data
                                            # Update timing with total time if available
                                            if first_token_received and metadata.get('total_time'):
                                                total_time_frontend = time.time() - request_sent_time
                                                network_overhead = total_time_frontend - metadata.get('total_time', 0)
                                                timing_placeholder.success(
                                                    f"⚡ **Time to first token:** `{time_to_first_token:.3f}s` (user sees)\n"
                                                    f"⏱️ **Total time:** `{total_time_frontend:.3f}s` (frontend) / `{metadata.get('total_time', 0):.3f}s` (backend only)\n\n"
                                                    f"📊 *Note: Frontend time includes network latency. Check terminal for backend-only timing.*"
                                                )
                                        
                                        elif data.get('type') == 'error':
                                            message_placeholder.error(f"Error: {data.get('message', 'Unknown error')}")
                                            timing_placeholder.empty()
                                            break
                                            
                                    except json.JSONDecodeError as e:
                                        # Skip malformed JSON
                                        continue
                        
                        # Remove typing cursor and show final response
                        if is_done:
                            message_placeholder.markdown(full_response)
                            # Hide timing placeholder after completion (optional - you can keep it if you want)
                            # timing_placeholder.empty()
                        
                        # Display metadata if available
                        if metadata:
                            with metadata_placeholder.expander("📎 Response Metadata"):
                                st.json(metadata)
                                
                                if metadata.get('video_url'):
                                    try:
                                        st.video(metadata['video_url'])
                                    except:
                                        st.markdown(f"Video URL: {metadata['video_url']}")
                                
                                if metadata.get('follow_up'):
                                    st.info(f"💡 Follow-up suggestion: {metadata['follow_up']}")
            else:
                # Standard (non-streaming) mode
                request_sent_time = time.time()
                timing_placeholder.info("⏳ Processing request (standard mode)...")
                
                standard_url = f"{api_url}/chat"
                params = {
                    "user_id": user_id,
                    "message": prompt
                }
                
                response = requests.get(standard_url, params=params, timeout=120)
                response_received_time = time.time()
                total_time_frontend = response_received_time - request_sent_time
                
                if response.status_code == 200:
                    data = response.json()
                    full_response = data.get('answer', '')
                    metadata = {
                        'intent': data.get('intent', ''),
                        'follow_up': data.get('follow_up'),
                        'video_url': data.get('video_url'),
                        'total_time': data.get('total_time', 0)
                    }
                    message_placeholder.markdown(full_response)
                    
                    # Show timing for standard mode
                    timing_placeholder.success(
                        f"⏱️ **Total response time:** `{total_time_frontend:.3f}s` (frontend) / "
                        f"`{metadata.get('total_time', 0):.3f}s` (backend)"
                    )
                    
                    with metadata_placeholder.expander("📎 Response Metadata"):
                        st.json(metadata)
                        if metadata.get('video_url'):
                            try:
                                st.video(metadata['video_url'])
                            except:
                                st.markdown(f"Video URL: {metadata['video_url']}")
                        if metadata.get('follow_up'):
                            st.info(f"💡 Follow-up suggestion: {metadata['follow_up']}")
                else:
                    message_placeholder.error(f"Error: {response.status_code} - {response.text}")
                    timing_placeholder.empty()
            
            # Add to message history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "metadata": metadata if metadata else None
            })
            
        except requests.exceptions.RequestException as e:
            message_placeholder.error(f"Connection error: {str(e)}")
            st.info("💡 Make sure the FastAPI server is running on the configured URL")
        except Exception as e:
            message_placeholder.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            st.session_state.streaming_active = False

# Stop streaming button
if st.session_state.streaming_active:
    if st.button("⏹️ Stop Streaming", key="stop"):
        st.session_state.streaming_active = False
        st.rerun()

# Clear chat button
if st.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Sidebar - Visualization options
st.sidebar.markdown("---")
st.sidebar.header("📊 Visualization Options")

show_explanation = st.sidebar.expander("ℹ️ How Streaming Works", expanded=False)
with show_explanation:
    st.markdown("""
    **Streaming Flow:**
    1. User sends question
    2. Backend processes (classification, retrieval) ~2-4s
    3. LLM starts generating → **First token arrives**
    4. Tokens stream in real-time → **UI updates progressively**
    5. Metadata arrives (video_url, follow_up)
    6. Stream completes
    
    **UI Features:**
    - ✅ Progressive text rendering (word-by-word)
    - ✅ Typing cursor indicator (▌)
    - ✅ Real-time markdown formatting
    - ✅ **Time to first token display** (from frontend submission)
    - ✅ Metadata display (video, follow-up)
    - ✅ Error handling
    
    **Performance:**
    - Standard mode: ~5-15s wait, then complete answer
    - Streaming mode: ~2-4s to first token, then progressive rendering
    - Timing shown includes: network latency + backend processing + LLM generation
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Frontend Changes Needed:**

1. **Replace fetch with SSE:**
```javascript
const eventSource = new EventSource('/chat-stream?...');
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'token') {
    appendToken(data.content);
  }
};
```

2. **Progressive Rendering:**
- Append tokens as they arrive
- Show typing cursor while streaming
- Update markdown renderer incrementally

3. **Metadata Handling:**
- Show video URL when metadata arrives
- Display follow-up question at end
""")

# Footer
st.markdown("---")
st.caption("💡 This is a demo visualization. The actual frontend would integrate similarly.")

