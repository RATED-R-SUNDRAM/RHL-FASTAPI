import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
import warnings, json
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings("ignore")
import logging


# Suppress PyTorch distributed / elastic warnings
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"   # hides torch C++ logs
os.environ["GLOG_minloglevel"] = "2"          # suppress GLOG warnings (used by torch)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
try:
    import torch
except RuntimeError:
    pass  # ignore torch internal runtime errors


# Suppress overly verbose loggers
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("pinecone").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)

# Optional: silence all third-party logs
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

import os, sys, warnings, logging

# Suppress Python warnings
warnings.filterwarnings("ignore")

# Suppress PyTorch / GLOG noise
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Suppress noisy loggers
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("streamlit").setLevel(logging.CRITICAL)
logging.getLogger("torch").setLevel(logging.ERROR)

# Swallow torch._classes RuntimeError bug on Windows
def silence_excepthook(exctype, value, traceback):
    if "torch._classes" in str(value):
        return
    sys.__excepthook__(exctype, value, traceback)

sys.excepthook = silence_excepthook


# ------------------ Init ------------------
def initialize_components():
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY2")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    llm = ChatOpenAI(
        model="gpt-5-nano-2025-08-07",
        api_key=openai_api_key,
        default_headers={"Authorization": f"Bearer {openai_api_key}"}
    )

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
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return llm, retriever

llm, retriever = initialize_components()

# ------------------ Session ------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

# ------------------ Profanity Filter ------------------
BAD_WORDS = ["fuck", "shit", "bitch", "asshole", "bastard", "slut"]
def contains_profanity(msg: str) -> bool:
    return any(bad in msg.lower() for bad in BAD_WORDS)

# ------------------ Prompts ------------------
from langchain.prompts import PromptTemplate

reformulate_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
Return output as JSON only: {{"Rewritten": "...", "Correction": "..."}}

### Rules

**Rewrite categorization & action:**
- **Medical term** → "give some information about <term> ""
- **New question** → keep as original question
- **Follow-up** → rewrite as a new standalone medical question, using chat_history for context,
     if last bot message suggested a follow-up, Use the most recent assistant message in chat_history contains 
  "Would you like to know about <X>?", and the user’s reply does not introduce new medical content, 
  rewrite full query related <X>. else if last bot message contains "Would you like to know about <X>?", and the user’s reply introduces new medical content, rewrite to accomodate last bot message and current follow-ups
   FOR eg :
            Last message of chat history : <answer > would you like to know about jaundice
            User  : yes
            Rewritten : Give information about jaundice and its cure
            Last message of chat history : <answer> would you like to know about malaria
            User : its prevention 
            Rewritten : Give information about prevention of malaria
            

- **Chitchat** → append "_chitchat" to rewritten
- **Profanity** → rewrite same input with "_chitchat", correction empty (chitchat pipeline will block)

**Correction rules:**
- Correction is ONLY meaningful if:
  - A spelling error changes interpretation (e.g., "johndice" → "jaundice")
  - An abbreviation is expanded (e.g., "IUFD" → "Intrauterine fetal death")
  - A word is incomplete/ambiguous and needs clarification (e.g., "treatmnt" → "treatment")
  - A misused word changes the medical meaning (e.g., "BP low sugar" → "low blood sugar")
- Correction is NOT meaningful if it’s only:
  - Adding/removing a question mark
  - Changing casing (bp → BP)
  - Fixing minor punctuation
  - Stylistic/grammar smoothing without changing meaning
- If no meaningful correction → set "Correction": ""

---

### Few-shot examples

1. 
chat_history: []
question: denger sign for johndice
Output: {{"Rewritten":"give some information about jaundice if cure available then cure","Correction":"I guess you meant 'danger sign for jaundice'"}}

2. 
chat_history: []
question: What is IUFD?
Output: {{"Rewritten":"What is Intrauterine fetal death?","Correction":"I guess you meant 'What is Intrauterine fetal death?'"}}

3. 
chat_history: [User: Tell me about fever | Bot: Fever is ... Would you like to know about symptoms?]
question: yes
Output: {{"Rewritten":"provide details on symptoms of fever","Correction":""}}

4. 
chat_history: [User: Tell me about fever | Bot: Fever is ...]
question: any treament?
Output: {{"Rewritten":"give some information about treatment of fever if cure available then cure","Correction":"I guess you meant 'any treatment?'"}} 

5. 
chat_history: []
question: hi there
Output: {{"Rewritten":"hi there_chitchat","Correction":""}}

6. 
chat_history: []
question: hey bitch
Output: {{"Rewritten":"hey bitch_chitchat","Correction":""}}

7. 
chat_history: [User: what is malaria? | Bot: Malaria info ... Would you like to know about prevention?]
question: sure
Output: {{"Rewritten":"provide details on prevention of malaria","Correction":""}}

---

chat_history: {chat_history}
question: {question}
JSON:
"""
)


from langchain.schema import HumanMessage

def medical_chatbot_pipeline(query, chat_history, retrieved_chunks, context_followup, main_llm, judge_llm):
    """
    Medical chatbot pipeline with:
    - Liberal judge to check sufficiency based on enough retreived context
    - Strict answer prompt template
    - Apology + alternative question handling
    """

    # Join retrieved chunks
    context = retrieved_chunks

    # ---- Judge sufficiency ----
    judge_prompt = f"""
    You are just a literal evaluator for a medical assistant system.
    You task is to judge if <context > is literally sufficient to answer the <query> 
    .

    Rules:
    - Reply in JSON: {{"sufficient": true/false, "reason": "...", "alternative": "possible related question strictly from context if possible else any another medical question"}}
    - "sufficient": false when the <context> has too many special characters and literally not possible to frame any response from <context>
    - "sufficient": true if an answer can be formed from context. 
    - "alternative": possible related question strictly from context if possible else any another medical question.
    For eg for sufficient : 
    IF query is about malaria , context is about jaundice -> return false
    If query is about cure of malaria , context is about malaria not adressing malaria -> return true


    User query: {query}
    Context:
    {context}
    """

    # ✅ Use invoke + HumanMessage
    decision_raw = judge_llm.invoke([HumanMessage(content=judge_prompt)]).content
    start = decision_raw.index("{")
    end = decision_raw.index("}") + 1
    try:
        decision = json.loads(decision_raw[start:end].strip())
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
       
    print("decision:",decision, type(decision))
    # ---- If sufficient: answer with strict rules ----
    if str(decision.get("sufficient")).lower()=='true':
        #print("I am answering")
        print("context_history",len(context_followup))
        answer_prompt = f"""
        You are a highly professional medical assistant. 

        Rules:
        - Always answer ONLY from <context> provided below.
        - Never use the web, external sources, or prior knowledge outside the given context.
        - Always follow below mentioned rules at all times :
            • Begin each answer with: **"According to <source>"** 
              (extract filename from metadata if available).
            • Answer concisely in bullet and sub-bullet points.
            • Keep under 150 words.
            • Do not copy-paste; summarize meaningfully.
            • Each bullet must be factual and context-bound.
        - If correction (from reformulation step) is provided, prepend it before bullets.
        - Respect chat history for coherence.
        - Always include a follow up question if <context_followup> is non-empty in the format(without bullet points) "Would you like to know about <a follow up question from context_followup not overlapping with the answer generated>?"


        Context:
        {context}

        User query:
        {query}

        Chat history:
        {chat_history}

        context_followup:
        {context_followup}

        Answer:
        """

        return main_llm.invoke([HumanMessage(content=answer_prompt)]).content

    # ---- If insufficient ----
    else:
        alt = decision['alternative']
        print("alt:",alt)
        if alt:
            return f"I apologize, but I do not have sufficient information to answer this question accurately. Would you like to know about {alt} instead?"
        return "I apologize, but I do not have sufficient information to answer this question accurately."

chitchat_prompt = PromptTemplate( input_variables=["conversation", "chat_history"],
 template=""" I am your cheerful bot 😃.
  Rules: - 
  - Always use professional words even if user has no professional words.
  - Reply to {conversation} in a friendly, chatty yet very professional tone respond with witty, empathetic tone.
  - Refrain from answering any off-topic questions, delegate to ask users to asking about medical questions
        For eg : 
             User : how are you doing ?
             Bot : Answer
             User : who is your favourite cricekter ?
             Bot : I am just a medical bot , i'll be useful if you ask me any medical questions. 
             User : hey bitch how are you
             Bot : Please keep it professional. I'm here to help you with medical related query .
  - Just maintain normal conversation basis<chat_history> dont search web or give external knowledge 
  - Limit response to max 40-50 words
  - For non-topic conversation in a fun way deflect the conversation towards asking users to move to asking medical questions.
  - End with: "Let’s dive deeper into <topic from chat_history or 'exciting medical topics'>!" 
  -If input contains absolute profanity, reply: "Whoa, let’s keep it polite, please! 😊" 
  Conversation: {conversation}
   Chat history: {chat_history} 
  Reply: """ )

# ------------------ Router ------------------
def route_intent(user_message: str):
    if not llm or not retriever:
        return "Error: System not fully initialized."

    history = st.session_state.history
    chat_history = "\n".join([f"User: {q} | Bot: {a}" for q, a, _ in history[-4:]]) if history else ""

    # Profanity pre-block
    if contains_profanity(user_message):
        answer = "Whoa, let’s keep it polite, please! 😊"
        st.session_state.history.append((user_message, answer, "dummy"))
        return answer

    # Reformulate intent
    reform = reformulate_prompt.format(chat_history=chat_history, question=user_message)
    rewritten_json = llm.invoke(reform).content.strip()
    print("reform", rewritten_json)
    try:
        parsed = json.loads(rewritten_json)
        rewritten = parsed.get("Rewritten", user_message)
        correction = parsed.get("Correction", "")
    except Exception:
        rewritten = user_message
        correction = ""

    # ---- Chitchat branch ----
    if rewritten.endswith("_chitchat"):
        clean_rewritten = rewritten.replace("_chitchat", "").strip()
        print("clean_rewritten: ",clean_rewritten)
        answer = llm.invoke(
            chitchat_prompt.format(conversation=clean_rewritten, chat_history=chat_history)
        ).content
        # FIX ✅: Add to history so UI shows it
        st.session_state.history.append((user_message, answer, "chitchat"))
        return answer

    # ---- Medical Q&A branch ----
    docs = retriever.invoke(rewritten)
    # Collect chunks
    chunks_with_source, chunks_without_source = [], []
    for d in docs:
        source = d.metadata.get("source")
        if source:
            source_name = os.path.splitext(os.path.basename(source))[0]
            chunks_with_source.append(f"From [{source_name}]: {d.page_content}")
        else:
            chunks_without_source.append(f"From [trusted medical reference]: {d.page_content}")

    selected_chunks = chunks_with_source if chunks_with_source else chunks_without_source

    # Build contexts
    context_answer = "\n\n".join(selected_chunks[:4])
    context_followup = "\n\n".join(selected_chunks[4:6] if len(selected_chunks) > 4 else [])

    print("context_answer==========", context_answer)
    print()
    print()
    print("context_followup===========", context_followup)
    print()

    answer = medical_chatbot_pipeline(rewritten,chat_history,context_answer,context_followup,llm,llm)
    if correction:
        answer = f"{correction}\n\n{answer}"

    # Save to session
    st.session_state.last_chunks = docs
    st.session_state.history.append((user_message, answer, "answer"))
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