import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone, ServerlessSpec


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

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# === Chat Model ===
llm2 = ChatOpenAI(temperature=0.2, model_name="gpt-4")

# === Session State ===
if "history" not in st.session_state:
    st.session_state.history = []

if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

# === Few-shot examples for intent detection ===
INTENT_EXAMPLES = [
    {"question": "What are the symptoms of depression?", "intent": "new_question"},
    {"question": "Can you explain more?", "intent": "follow_up_chunks"},
    {"question": "What do you mean by that?", "intent": "clarification"},
    {"question": "Thanks!", "intent": "gratitude"},
    {"question": "Tell me a joke", "intent": "chitchat"},
    {"question": "What causes anxiety?", "intent": "new_question"},
    {"question": "How does that compare to bipolar disorder?", "intent": "follow_up_reretrieve"}
]

intent_prompt = PromptTemplate.from_template(
    """
You are an intent classifier for a medical chatbot. Classify the user's message into one of the following:
- new_question: Asking a fresh medical question.
- follow_up_chunks: Wants more info based on the previous answer.
- follow_up_reretrieve: Asks a follow-up requiring deeper search.
- clarification: Clarifies something from the previous answer.
- gratitude: Thanks the assistant.
- chitchat: Casual conversation.

Output one label: new_question, follow_up_chunks, follow_up_reretrieve, clarification, gratitude, or chitchat.

Examples:
{% for ex in examples %}- {{ ex.question }} → {{ ex.intent }}
{% endfor %}

User: {input}
Intent:
"""
)
def build_intent_prompt(user_input: str, examples: list[dict]) -> str:
    formatted_examples = "\n".join([f"- {ex['question']} → {ex['intent']}" for ex in examples])
    return f"""
You are an intent classifier for a medical chatbot. Classify the user's message into one of the following basis existing recent communication history:
- new_question: Asking a fresh medical question.
- follow_up_chunks: Wants more info based on the previous answer.
- follow_up_reretrieve: Asks a follow-up requiring deeper search.
- gratitude: Thanks the assistant.
- chitchat: Casual conversation.

Output one label: new_question, follow_up_chunks, follow_up_reretrieve, gratitude, or chitchat.

communication history:
{st.session_state.history[:10]}
Examples:
{formatted_examples}

User: {user_input}
Intent:"""


def detect_intent(message: str) -> str:
    prompt = build_intent_prompt(message, INTENT_EXAMPLES)
    result = llm.invoke(prompt).content.strip().lower()
    return result

# === Prompt templates ===
answer_prompt_template = PromptTemplate.from_template(
    """
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

Question: {question}
Answer:
"""
)

clarify_prompt = PromptTemplate.from_template(
    """
You previously answered:
{last_answer}

User follow-up:
{question}

Clarify or expand based on your last answer.
"""
)

reformulate_prompt = PromptTemplate.from_template(
    """
You are a helpful assistant. Given a follow-up question and the last Q&A, rewrite the follow-up into a standalone question.

Previous Q&A:
Q: {last_question}
A: {last_answer}

Follow-up: {question}

Rewritten Question:
"""
)

def route_intent(user_message: str):
    intent = detect_intent(user_message)
    history = st.session_state.history

    if intent == "new_question":
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        docs = retriever.invoke(user_message)
        chunks = []
        for d in docs:
            src = os.path.basename(d.metadata.get("source", ""))
            chunks.append(f"From [{src}]: {d.page_content}")
        context ="\n\n".join(chunks)
        prompt = answer_prompt_template.format(context=context, question=user_message)
        answer = llm.invoke(prompt).content
        st.session_state.last_chunks = docs

    elif intent == "follow_up_chunks":
        chunks = st.session_state.last_chunks
        context = "\n\n".join([doc.page_content for doc in chunks])
        prompt = answer_prompt_template.format(context=context, question=user_message)
        answer = llm.invoke(prompt).content

    elif intent == "follow_up_reretrieve":
        last_turns = [m for m in reversed(history) if m[2] in ["new_question", "follow_up_reretrieve"]]
        if last_turns:
            last_question, last_answer, _ = last_turns[0]
            reform = reformulate_prompt.format(
                last_question=last_question,
                last_answer=last_answer,
                question=user_message
            )
            rewritten = llm.invoke(reform).content
            #docs = retriever.get_relevant_documents()
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            docs = retriever.invoke(rewritten)
            chunks = []
            for d in docs:
                src = os.path.basename(d.metadata.get("source", ""))
                chunks.append(f"From [{src}]: {d.page_content}")
            context ="\n\n".join(chunks)
            
            prompt = answer_prompt_template.format(context=context, question=rewritten)
            answer = llm.invoke(prompt).content
            st.session_state.last_chunks = docs
        else:
            answer = "Sorry, I couldn't find enough context to answer."

    elif intent == "clarification":
        last_turn = [m for m in reversed(history) if m[2] in ["new_question", "follow_up_reretrieve", "follow_up_chunks"]]
        if last_turn:
            last_question, last_answer, _ = last_turn[0]
            prompt = clarify_prompt.format(last_answer=last_answer, question=user_message)
            answer = llm.invoke(prompt).content
        else:
            answer = "Sorry, I need more context to clarify."

    elif intent == "gratitude":
        answer = "You're welcome! Let me know if you have more questions."

    elif intent == "chitchat":
        answer = "I'm here to help with medical questions. Feel free to ask one!"

    else:
        answer = "Sorry, I couldn't understand that. Please rephrase."

    st.session_state.history.append((user_message, answer, intent))
    return answer

# === Streamlit UI ===
st.set_page_config(page_title="Medical Chatbot", layout="centered")
st.title("🩺 Smart Medical Chatbot")

user_input = st.chat_input("Ask a medical question...")

if user_input:
    with st.spinner("Thinking..."):
        response = route_intent(user_input)

for i, (q, a, intent) in enumerate(reversed(st.session_state.history)):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(f"**({intent})**\n{a}")
