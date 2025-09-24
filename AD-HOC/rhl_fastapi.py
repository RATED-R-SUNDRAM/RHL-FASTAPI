import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
import aiosqlite
from typing import List, Tuple

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY2")
openai_api_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()

# Initialize components
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=openai_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "chat-models-v1-all-minilm-l6"
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, metric="cosine", dimension=384, spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embedding_model, pinecone_api_key=pinecone_api_key)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Init error: {e}")

# Database setup
async def init_db():
    async with aiosqlite.connect("chat_history.db") as db:
        await db.execute("CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY, user_id TEXT, question TEXT, answer TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        await db.commit()

# Data access
async def get_history(user_id: str) -> List[Tuple[str, str]]:
    async with aiosqlite.connect("chat_history.db") as db:
        cursor = await db.execute("SELECT question, answer FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5", (user_id,))
        rows = await cursor.fetchall()
        return [(row[0], row[1]) for row in rows]

async def save_history(user_id: str, question: str, answer: str):
    async with aiosqlite.connect("chat_history.db") as db:
        await db.execute("INSERT INTO chat_history (user_id, question, answer) VALUES (?, ?, ?)", (user_id, question, answer))
        await db.commit()

# Prompts
reformulate_prompt = PromptTemplate.from_template(
    """
You are a smart medical assistant for a RAG chatbot.

Tasks:
1. Correct spelling mistakes in the input prompt, assuming a medical context.
2. Rephrase any medical abbreviations to their full terms (e.g., IUFD to Intrauterine fetal death).
3. Categorize and act:
   - If only medical terminology, rephrase to "cure of [term]."
   - If a new medical question, return as is.
   - If a follow-up, rewrite based on last Q&A and context for document retrieval.
   - If chitchat, return "chitchat."

Previous Q&A:
Q: {last_question}
A: {last_answer}

Follow-up: {question}

Rewritten Question:
"""
)

answer_prompt_template = PromptTemplate.from_template(
   """ You are a medical assistant. Answer only using the given context. If the answer is not in the context, reply:
"I apologize, but I do not have sufficient information in my documents to answer this question accurately."

Start with: **"According to <source>.pdf"**, where <source> is from metadata.
Do not use general knowledge or suggest external searches.

Guidelines:
- Be concise, medically accurate.
- Use context to address unanswered aspects from the previous answer.
- Consider recent chat history for coherence.
- Word limit: 150 words.
- Suggest a relevant follow-up question (e.g., "Would you like to know about...").

Context:
{context}
Previous Answer:
{prev_answer}
Recent History (last turn):
{prev_history}

Question: {question}
Answer:
"""
)

# Logic
async def route_intent(user_id: str, user_message: str):
    if not llm or not retriever:
        raise HTTPException(status_code=500, detail="System not initialized.")
    history = await get_history(user_id)
    last_turn = history[-1] if history else None
    prev_question, prev_answer = ("", "") if last_turn is None else last_turn

    reform = reformulate_prompt.format(last_question=prev_question, last_answer=prev_answer, question=user_message)
    rewritten = llm.invoke(reform).content.strip()
    if rewritten.lower() == "chitchat":
        answer = "I'm here for medical questions. Ask one!"
    else:
        docs = retriever.invoke(rewritten)
        chunks = [f"From [{os.path.basename(d.metadata.get('source', ''))}.pdf]: {d.page_content}" for d in docs]
        context = "\n\n".join(chunks[:3])
        prompt = answer_prompt_template.format(context=context, prev_answer=prev_answer, prev_history=f"User: {prev_question} | Bot: {prev_answer}" if last_turn else "", question=rewritten)
        answer = llm.invoke(prompt).content
        await save_history(user_id, user_message, answer)
    return {"answer": answer, "history": history + [(user_message, answer)]}

# API
@app.on_event("startup")
async def startup():
    await init_db()

@app.post("/chat")
@app.get("/chat")
async def chat(user_id: str, message: str):
    return await route_intent(user_id, message)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)