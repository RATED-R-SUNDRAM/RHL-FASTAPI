"""
Standalone Streamlit application with integrated FastAPI backend.
This includes both frontend and backend - no separate API server needed!

DEPLOYMENT:
1. Make sure rhl_fastapi_deploy.py is in the same directory
2. Run: streamlit run streamlit_standalone.py
3. That's it! Everything runs in one process.
"""
import streamlit as st
import asyncio
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Import backend module
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    import rhl_fastapi_deploy as backend
    from rhl_fastapi_deploy import (
        init_db,
        get_history,
        save_history,
        get_chat_context,
        classify_and_reformulate,
        handle_chitchat,
        medical_pipeline_api,
        synthesize_answer_stream,
        hybrid_retrieve,
        bert_filter_candidates,
    )
    from fastapi import BackgroundTasks
except ImportError as e:
    st.error(f"❌ Error importing backend: {e}")
    st.error("Make sure `rhl_fastapi_deploy.py` is in the same directory!")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Medical Chatbot - Standalone",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏥 Medical Chatbot - Standalone")
st.markdown("**✨ Integrated Frontend + Backend - Single deployment!**")

# Sidebar
st.sidebar.header("⚙️ Configuration")
user_id = st.sidebar.text_input("User ID", value="demo_user")
stream_mode = st.sidebar.radio("Response Mode", ["Streaming", "Standard"], index=0)

# Initialize backend models (cached - runs only once)
@st.cache_resource
def init_backend_models():
    """Initialize all backend models. This is cached and runs only once."""
    import asyncio
    from dotenv import load_dotenv
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from chromadb.config import Settings
    import chromadb
    from pathlib import Path
    
    load_dotenv()
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    try:
        from flashrank import Ranker
        FLASHRANK_AVAILABLE = True
    except:
        FLASHRANK_AVAILABLE = False
        Ranker = None
    
    try:
        # Initialize database
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(init_db())
        
        # Initialize embedding model
        backend.embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        backend.EMBED_DIM = backend.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize reranker
        try:
            if FLASHRANK_AVAILABLE and Ranker:
                backend.reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
            else:
                backend.reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        except:
            backend.reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        
        # Initialize ChromaDB
        VECTORSTORE_DIR = Path("FILES/local_vectorstore")
        if not VECTORSTORE_DIR.exists():
            raise RuntimeError(f"ChromaDB vector store not found at {VECTORSTORE_DIR}. Run setup_local_vectorstore.py first!")
        
        client = chromadb.PersistentClient(
            path=str(VECTORSTORE_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        backend.chromadb_collection = client.get_collection(name="medical_documents")
        
        # Initialize LLMs
        backend.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, api_key=OPENAI_API_KEY)
        backend.summarizer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
        backend.reformulate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
        backend.classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
        backend.gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=GOOGLE_API_KEY)
        
        # Initialize cache and video systems
        from rhl_fastapi_deploy import CacheSystem, VideoMatchingSystem
        backend.cache_system = CacheSystem()
        backend.video_system = VideoMatchingSystem()
        
        loop.close()
        return True
    except Exception as e:
        st.error(f"Initialization error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False

# Check if backend is initialized
if 'backend_ready' not in st.session_state:
    with st.spinner("🔄 Initializing backend models (this may take 30-60 seconds)..."):
        if init_backend_models():
            st.session_state.backend_ready = True
            st.success("✅ Backend initialized successfully!")
        else:
            st.error("❌ Backend initialization failed. Check errors above.")
            st.stop()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'streaming_active' not in st.session_state:
    st.session_state.streaming_active = False

# Display chat history
st.header("💬 Chat Interface")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg and msg["metadata"]:
            with st.expander("📎 Response Metadata"):
                st.json(msg["metadata"])
                if msg["metadata"].get('video_url'):
                    try:
                        st.video(msg["metadata"]['video_url'])
                    except:
                        st.markdown(f"Video URL: {msg['metadata']['video_url']}")
                if msg["metadata"].get('follow_up'):
                    st.info(f"💡 Follow-up: {msg['metadata']['follow_up']}")

# Helper to run async pipeline
async def process_message_streaming(user_id: str, message: str):
    """Process message with streaming support."""
    background_tasks = BackgroundTasks()
    
    # Run the full pipeline
    result = await medical_pipeline_api(user_id, message, background_tasks)
    
    return result

async def process_message_standard(user_id: str, message: str):
    """Process message in standard mode."""
    background_tasks = BackgroundTasks()
    result = await medical_pipeline_api(user_id, message, background_tasks)
    return result

# Chat input
if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        timing_placeholder = st.empty()
        message_placeholder = st.empty()
        metadata_placeholder = st.empty()
        full_response = ""
        metadata = None
        
        try:
            request_start = time.time()
            timing_placeholder.info("⏳ Processing request...")
            
            # Run async pipeline
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                if stream_mode == "Streaming":
                    # For streaming, we'll use the standard pipeline but display progressively
                    # Note: True streaming requires the streaming endpoint, but for standalone
                    # we'll simulate it with progressive display
                    result = await process_message_streaming(user_id, prompt)
                else:
                    result = await process_message_standard(user_id, prompt)
                
                total_time = time.time() - request_start
                full_response = result.get('answer', '')
                metadata = {
                    'intent': result.get('intent', ''),
                    'follow_up': result.get('follow_up'),
                    'video_url': result.get('video_url'),
                    'total_time': result.get('total_time', total_time)
                }
                
                # Display response
                if stream_mode == "Streaming":
                    # Simulate streaming by displaying word by word
                    words = full_response.split()
                    displayed = ""
                    first_token_time = None
                    
                    for i, word in enumerate(words):
                        if i == 0:
                            first_token_time = time.time()
                            time_to_first = first_token_time - request_start
                            timing_placeholder.success(
                                f"⚡ **Time to first token:** `{time_to_first:.3f}s`\n"
                                f"⏱️ **Processing time:** `{time_to_first:.3f}s`"
                            )
                        
                        displayed += word + " "
                        message_placeholder.markdown(displayed + "▌")
                        time.sleep(0.02)  # Small delay for streaming effect
                    
                    message_placeholder.markdown(displayed)
                else:
                    message_placeholder.markdown(full_response)
                    timing_placeholder.success(
                        f"⏱️ **Total time:** `{total_time:.3f}s`"
                    )
                
                # Display metadata
                if metadata:
                    with metadata_placeholder.expander("📎 Response Metadata"):
                        st.json(metadata)
                        if metadata.get('video_url'):
                            try:
                                st.video(metadata['video_url'])
                            except:
                                st.markdown(f"Video URL: {metadata['video_url']}")
                        if metadata.get('follow_up'):
                            st.info(f"💡 Follow-up: {metadata['follow_up']}")
                
            finally:
                loop.close()
            
            # Add to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "metadata": metadata if metadata else None
            })
            
        except Exception as e:
            message_placeholder.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

# Clear chat button
if st.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.success("""
**Standalone Deployment**

✅ No separate FastAPI server needed
✅ Direct function calls (faster!)
✅ Single command to deploy

**Backend:** rhl_fastapi_deploy.py
**Status:** Ready
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**To Deploy:**

```bash
streamlit run streamlit_standalone.py
```

That's it! Everything runs in one process.
""")

# Footer
st.markdown("---")
st.caption("💡 Standalone deployment - All backend logic runs within Streamlit process")
