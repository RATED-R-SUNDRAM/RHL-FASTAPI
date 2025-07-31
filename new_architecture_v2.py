import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY2")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Session State Initialization
if "history" not in st.session_state:
    st.session_state.history = []
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

# Embeddings and LLM
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # Explicitly set device
    )
except Exception as e:
    st.error(f"Error initializing embeddings: {e}. Please ensure 'sentence-transformers' is installed with `pip install sentence-transformers` and compatible PyTorch version.")
    embedding_model = None

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=openai_api_key,
    default_headers={"Authorization": f"Bearer {openai_api_key}"}
)

# Pinecone Initialization
pc = Pinecone(api_key=pinecone_api_key)
index_name = "chat-models-v1-all-minilm-l6"
embedding_dim = len(embedding_model.embed_query("test")) if embedding_model else 384  # Default dim if embedding fails

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        metric="cosine",
        dimension=embedding_dim,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

vector_store = PineconeVectorStore(
    index_name=index_name, embedding=embedding_model, pinecone_api_key=pinecone_api_key
) if embedding_model else None
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6}) if vector_store else None

# Medical Abbreviations
MEDICAL_ABBRS = {
    "IUFD": "Intrauterine fetal death",
    "PROM": "Prelabour Rupture of Membranes"
}

# Intent Examples
INTENT_EXAMPLES = [
    {"question": "What are symptoms?", "intent": "new_question"},
    {"question": "Explain more", "intent": "follow_up"},
    {"question": "What about diabetes?", "intent": "follow_up"},
    {"question": "Thanks!", "intent": "gratitude"},
    {"question": "How are you?", "intent": "chitchat"}
]

intent_prompt = PromptTemplate.from_template(
    """
You are an intent classifier for a medical chatbot. Classify the user's message into:
- new_question: A fresh medical query.
- follow_up: Seeks elaboration or new info based on prior context.
- gratitude: Thanks or positive feedback.
- chitchat: Casual talk.

Output one label: new_question, follow_up, gratitude, or chitchat.

Examples:
{% for ex in examples %}- {{ ex.question }} → {{ ex.intent }}
{% endfor %}

User: {input}
Intent:
"""
)

def build_intent_prompt(user_input: str, examples: list[dict]) -> str:
    formatted_examples = "\n".join([f"- {ex['question']} → {ex['intent']}" for ex in examples])
    history = "\n".join([f"User: {q} | Bot: {a}" for q, a, _ in st.session_state.history[-2:]])
    return f"""
You are an intent classifier. Classify based on recent history:
- new_question: Fresh query.
- follow_up: Elaboration or new info from prior context.
- gratitude: Thanks.
- chitchat: Casual talk.

History:
{history}
Examples:
{formatted_examples}

User: {user_input}
Intent:"""

def detect_intent(message: str) -> str:
    if not llm:
        return "error"
    prompt = build_intent_prompt(message, INTENT_EXAMPLES)
    return llm.invoke(prompt).content.strip().lower()

# Reformulation Prompt
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

# Answer Prompt
answer_prompt_template = PromptTemplate.from_template(
    """
You are a medical assistant. Answer only using the given context. If the answer is not in the context, reply:
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

def route_intent(user_message: str):
    print("here 1")
    if not llm or not retriever:
        return "Error: System not fully initialized. Check dependencies."
    print("here 2")
    history = st.session_state.history
    last_turn = history[-1] if history else None
    prev_question, prev_answer, _ = last_turn if last_turn else ("", "", "")
    prev_chunks = st.session_state.last_chunks if st.session_state.last_chunks else []
    
    reform = reformulate_prompt.format(last_question=prev_question, last_answer=prev_answer, question=user_message)
    rewritten = llm.invoke(reform).content.strip()
    print(rewritten)
    if rewritten.lower() == "chitchat":
        
        answer= "I'm here for medical questions. Ask one!"
    else:
        docs = retriever.invoke(rewritten)
        chunks = [f"From [{os.path.basename(d.metadata.get('source', ''))}.pdf]: {d.page_content}" for d in docs]
        context = "\n\n".join(chunks[:3])  # Top 3 for brevity
        prompt = answer_prompt_template.format(context=context, prev_answer=prev_answer, prev_history=f"User: {prev_question} | Bot: {prev_answer}" if last_turn else "", question=rewritten)
        answer = llm.invoke(prompt).content
        st.session_state.last_chunks = docs
        #st.session_state.history.append((user_message, answer, "follow_up" if last_turn else "new_question"))
    st.session_state.history.append((user_message, answer, "dummy"))
    return answer

# Streamlit UI
st.set_page_config(page_title="Medical Chatbot", layout="centered")
st.title("🩺 Smart Medical Chatbot")

user_input = st.chat_input("Ask a medical question...")

if user_input:
    with st.spinner("Thinking..."):
        response = route_intent(user_input)
        #print(response)
# Safely handle history display
if st.session_state.history:
    for i, (q, a, intent) in enumerate(reversed(st.session_state.history)):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(f"{a}")