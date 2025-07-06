from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# ========== ENV SETUP ==========
load_dotenv()
pinecone_api_key = st.secrets["PINECONE_API_KEY2"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

# ========== EMBEDDINGS & LLM ==========
embedding_hf = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

medical_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=openai_api_key,
    default_headers={"Authorization": f"Bearer {openai_api_key}"}
)

# ========== PINECONE INIT ==========
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

# ========== PROMPT (Claude-style grounded) ==========
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

# ========== CONTEXT RETRIEVAL ==========
def get_context(query):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    return "\n\n".join(a.page_content for a in docs)

# ========== STREAMLIT UI ==========
st.title("Medical Chatbot")
st.write("Ask your medical or general questions below.")

# Session chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# On new question
if prompt := st.chat_input("Enter your query:"):

    # Append current user question
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # === Trim chat history to last 2 Q&A pairs (i.e. 4 messages: user, assistant, user, assistant)
    trimmed = st.session_state.chat_history[-4:]
    chat_str = ""
    for m in trimmed:
        role_prefix = "User:" if m["role"] == "user" else "Assistant:"
        chat_str += f"{role_prefix} {m['content']}\n"

    # === Build input and invoke chain ===
    context = get_context(prompt)
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: get_context(x["question"]),
            chat_history=lambda x: chat_str
        )
        | medical_prompt
        | medical_llm
    )
    response = chain.invoke({
        "question": prompt
    })

    # Append assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": response.content})
    with st.chat_message("assistant"):
        st.write(response.content)
