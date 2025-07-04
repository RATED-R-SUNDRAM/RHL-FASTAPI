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

"LOADING ENV VARIABLE"
load_dotenv()
pinecone_api_key = st.secrets["PINECONE_API_KEY2"]
openai_api_key = st.secrets["OPENAI_API_KEY"]
"LOADING EMBEDDING AND LLM"
embedding_hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

medical_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=openai_api_key,
    default_headers={"Authorization": f"Bearer {openai_api_key}"}
)

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

"INITIALIZING VECTOR STORE"
vector_store = PineconeVectorStore(index_name=index_name, embedding=embedding_hf, pinecone_api_key=pinecone_api_key)

medical_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful and knowledgeable medical assistant.

Your task is to provide concise, factual answers based strictly on the information provided in the context below.

Guidelines for Response Style:
- Write your answer as if it’s general knowledge from a human expert, without mentioning documents, context, or sources.
- Do not reference, quote, or mention the context or any documents in your answer.
- Do not include phrases like “Based on the context…” or “The document states…”.
- Summarize, paraphrase, and explain naturally as if speaking directly to the user.
- Stay within 150-200 words.
- If the information needed is missing, simply respond: "The Question is out of scope of this Application."

Context:
{context}

User Question:
{question}

Your Answer:

"""
)

def get_context(query):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    return "\n\n".join(a.page_content for a in docs)

"LOADING CHATBOT"

""" STREAMLIT UI """
st.title("Medical Chatbot")
st.write("Ask your medical or general questions below. ")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input field for user query
if prompt := st.chat_input("Enter your query:"):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get response
    context = get_context(prompt)
    chain = (
        RunnablePassthrough.assign(context=lambda x: get_context(x["question"]))
        | medical_prompt
        | medical_llm
    )
    response = chain.invoke({"question": prompt})
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response.content})
    with st.chat_message("assistant"):
        st.write(response.content)