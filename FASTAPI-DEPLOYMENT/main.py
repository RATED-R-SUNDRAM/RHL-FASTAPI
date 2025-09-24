from fastapi import FastAPI, Request
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY2")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize app and embeddings
app = FastAPI()
embedding_hf = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Initialize LLM
medical_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=openai_api_key,
    default_headers={"Authorization": f"Bearer {openai_api_key}"}
)

# Pinecone setup
pc = Pinecone(api_key=pinecone_api_key)
embedding_dimension = len(embedding_hf.embed_query("test"))
index_name = "chat-models-v1-all-minilm-l6"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        metric="cosine",
        dimension=embedding_dimension,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

vector_store = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding_hf,
    pinecone_api_key=pinecone_api_key
)

# In-memory chat history store
chat_histories = {}

# Prompt Template
medical_prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""
You are an AI medical assistant that responds **only using the given context**.
Never guess, generalize, or use outside knowledge.

Instructions:
- Only answer if the retrieved context contains enough relevant info.
- If not, respond exactly: "The question is out of scope of this application."
- NEVER mention the context, documents, retrieval, or sources.
- NEVER say phrases like "based on the context..." or "from the document...".
- Be concise, medically accurate, and direct.
- Maximum 150 words.
- You may include the most recent prior exchanges if relevant.

Context:
{context}

Recent Conversation:
{chat_history}

Current User Question:
{question}

Your Answer:
"""
)

# Context Retriever
def get_context(query):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    return "\n\n".join(a.page_content for a in docs)

# API routes
@app.get("/")
def root():
    return {"message": "RAG API is running"}

@app.get("/bot/{query}")
def get_bot_response(query: str, request: Request, word_limit: int = 150):
    user_id = request.query_params.get("user_id", "test_user")

    # Get and trim history to last 2 Q&A pairs (max 4 messages)
    history = chat_histories.get(user_id, [])
    if len(history) > 4:
        history = history[-4:]
    
    # Format history with roles
    formatted_history = ""
    for i, msg in enumerate(history):
        role = "User:" if i % 2 == 0 else "Assistant:"
        formatted_history += f"{role} {msg}\n" if i < len(history) - 1 or i % 2 == 0 else f"{role} {msg}"

    # Update history with new query
    history.append(query)
    chat_histories[user_id] = history

    # Retrieve context
    context = get_context(query)

    # Prepare chain
    chain = (
        RunnablePassthrough.assign(context=lambda x: context)
        | medical_prompt
        | medical_llm
    )

    result = chain.invoke({
        "question": query,
        "chat_history": formatted_history,
    })

    return {"response": result.content.replace("\n\n", "\n").replace("\\n", "\n")}