import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone, ServerlessSpec

# ========== ENV SETUP ==========
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY2")
openai_api_key = os.getenv("OPENAI_API_KEY")

# ========== EMBEDDINGS & LLM ==========
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=openai_api_key,
    default_headers={"Authorization": f"Bearer {openai_api_key}"}
)

# ========== PINECONE INIT ==========
pc = Pinecone(api_key=pinecone_api_key)
index_name = "chat-models-v1-all-minilm-l6"
embedding_dim = len(embedding_model.embed_query("test"))

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        metric="cosine",
        dimension=embedding_dim,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

vector_store = PineconeVectorStore(
    index_name=index_name, embedding=embedding_model, pinecone_api_key=pinecone_api_key
)

# ========== REWRITER ==========
rewrite_prompt = PromptTemplate(
    input_variables=["question", "chat_history", "context"],
    template="""
You are a smart assistant. 
Rewrite the user question only if it is a follow-up (like: "give more info", "summarize", "what do you mean") using the chat history and document context.
If the current question makes sense on its own, return it unchanged.

Chat History:
{chat_history}

Relevant Document Chunks:
{context}

User's Question:
{question}

Rewritten Query:
"""
)

def rewrite_query(question, chat_history, context):
    history_str = ""
    for m in chat_history:
        prefix = "User:" if m["role"] == "user" else "Assistant:"
        history_str += f"{prefix} {m['content']}\n"

    chain = rewrite_prompt | llm
    rewritten = chain.invoke({"question": question, "chat_history": history_str, "context": context})
    return rewritten.content.strip()

# ========== MAIN PROMPT ==========
main_prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""
You are a medical assistant. Only answer using the given context. If the answer is not present, reply:
"The question is out of scope of this application."

Start your answer with: **"According to <source>.pdf"**, where <source> is the file name from metadata.
Do not use general knowledge or web data.

Guidelines:
- Be concise and medically accurate.
- Use only retrieved chunks for the answer.
- Word limit: 150 words.
- Include document name (from metadata) for citation.

Context:
{context}

Recent Conversation:
{chat_history}

User Question:
{question}

Answer:
"""
)

# ========== CONTEXT RETRIEVAL ==========
def get_context(query):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    chunks = []
    for d in docs:
        src = os.path.basename(d.metadata.get("source", ""))
        chunks.append(f"From [{src}]: {d.page_content}")
    return "\n\n".join(chunks)

# ========== STREAMLIT UI ==========
st.title("🩺 Medical Chatbot (Document-grounded)")
#st.write("Ask questions grounded in uploaded PDFs. No external info is used.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt_input := st.chat_input("Ask something..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Trim history to last 15 messages
    trimmed_history = st.session_state.chat_history[-15:-1]  # exclude current message

    # Fetch context early so rewriter uses it too
    context = get_context(prompt_input)

    # Rewriting
    rewritten_query = rewrite_query(prompt_input, trimmed_history, context)

    # Build prompt
    short_history = ""
    for m in trimmed_history:
        prefix = "User:" if m["role"] == "user" else "Assistant:"
        short_history += f"{prefix} {m['content']}\n"

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: context,
            chat_history=lambda x: short_history
        )
        | main_prompt
        | llm
    )
    result = chain.invoke({"question": rewritten_query})
    final_response = result.content.strip()

    st.session_state.chat_history.append({"role": "assistant", "content": final_response})
    with st.chat_message("assistant"):
        st.markdown(final_response)
