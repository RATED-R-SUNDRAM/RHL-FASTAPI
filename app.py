from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY2")
openai_api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

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

vector_store = PineconeVectorStore(index_name=index_name, embedding=embedding_hf, pinecone_api_key=pinecone_api_key)

medical_prompt = PromptTemplate(
    input_variables=["context", "question", "word_limit"],
    template="""You are a highly factual, document-grounded AI assistant. 

Your task:
- Use only the provided context below.
- Do not include external knowledge or assumptions.
- Answer the user question clearly and concisely.
- Strictly stay within {word_limit} words.
- If the context does not contain enough information, reply: "Insufficient context to answer."

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

@app.get("/")
def root():
    return {"message": "RAG API is running"}

@app.get("/bot/")
def get_bot_response(query: str, word_limit: int):
    chain = (
        RunnablePassthrough.assign(context=lambda x: get_context(x["question"]))
        | medical_prompt
        | medical_llm
    )

    result = chain.invoke({
        "question": query,
        "word_limit": word_limit
    })

    return {"response": result.content}
