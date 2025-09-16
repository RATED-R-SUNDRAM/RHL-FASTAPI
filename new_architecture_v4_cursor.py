import os
import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import numpy as np


from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except Exception:
    CROSS_ENCODER_AVAILABLE = False

from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI  # your existing wrapper used earlier

from sentence_transformers import CrossEncoder

# Load cross-encoder once
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# -------------------- CONFIG --------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s • %(levelname)s • %(message)s")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY_NEW")
INDEX_NAME = os.getenv("PINECONE_INDEX", "medical-chatbot-index")

if not OPENAI_API_KEY or not PINECONE_KEY:
    raise RuntimeError("OPENAI_API_KEY or PINECONE_API_KEY_NEW missing in environment")


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # optional (fastish)

# -------------------- MODELS & CLIENTS --------------------
logging.info("Loading embedding model...")
# embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# EMBED_DIM = embedding_model.get_sentence_embedding_dimension()

logging.info("Pinecone init...")
# pc = Pinecone(api_key=PINECONE_KEY)
# if INDEX_NAME not in pc.list_indexes().names():
#     logging.info("Creating Pinecone index (if needed)...")
#     pc.create_index(name=INDEX_NAME, dimension=EMBED_DIM, metric="cosine",
#                     spec=ServerlessSpec(cloud="aws", region="us-east-1"))
# index = pc.Index(INDEX_NAME)
# logging.info("Pinecone index ready.")

logging.info("Initializing LLM clients...")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
chitchat_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=OPENAI_API_KEY)
summarizer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)

# if CROSS_ENCODER_AVAILABLE:
#     try:
#         logging.info("Loading cross-encoder for reranking...")
#         cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
#     except Exception as e:
#         logging.warning("Unable to initialize CrossEncoder: %s", e)
#         CROSS_ENCODER_AVAILABLE = False
#         cross_encoder = None
# else:
#     cross_encoder = None
#     logging.info("CrossEncoder not available - falling back to vector+BM25 fusion only.")

# -------------------- CHAT HISTORY MANAGEMENT --------------------
MAX_VERBATIM_PAIRS = 3  # keep last 3 Q-A pairs verbatim
def init_session():
    if "history_pairs" not in st.session_state:
        st.session_state.history_pairs = []  # list[(q,a,intent)]
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "last_suggested" not in st.session_state:
        st.session_state.last_suggested = ""
    if "debug" not in st.session_state:
        st.session_state.debug = []
    if "pinecone_index" not in st.session_state:
        pc = Pinecone(api_key=PINECONE_KEY)
        index = pc.Index(INDEX_NAME)
        st.session_state.pinecone_index = index

    if "reranker" not in st.session_state:
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        st.session_state.reranker = cross_encoder
    if "embedding_model" not in st.session_state:
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        EMBED_DIM = embedding_model.get_sentence_embedding_dimension()
        st.session_state.embedding_model = embedding_model
        st.session_state.EMBED_DIM= EMBED_DIM

def append_debug(msg):
    logging.info(msg)
    st.session_state.debug.append(msg)
    # keep debug size bounded
    if len(st.session_state.debug) > 200:
        st.session_state.debug = st.session_state.debug[-200:]

def update_chat_history(user_message, bot_reply, intent):
    st.session_state.history_pairs.append((user_message, bot_reply, intent))
    # if we exceed MAX_VERBATIM_PAIRS, summarize older
    if len(st.session_state.history_pairs) > MAX_VERBATIM_PAIRS:
        # old entries to compress
        old = st.session_state.history_pairs[:-MAX_VERBATIM_PAIRS]
        text = "\n".join([f"User: {q} | Bot: {a}" for q, a, _ in old])
        prompt = f"Summarize the following dialogues in one concise medical-context sentence so the bot can have context about the discussions happened so far(for future context):\n\n{text}"
        try:
            summary = summarizer_llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception:
            summary = ""
        st.session_state.summary = (st.session_state.summary + " " + summary).strip()
        st.session_state.history_pairs = st.session_state.history_pairs[-MAX_VERBATIM_PAIRS:]

def get_chat_context():
    # Filter out chitchat messages from verbatim history when constructing context for LLM
    filtered_verbatim_pairs = [(q, a) for q, a, intent in st.session_state.history_pairs if intent != "chitchat"]
    verbatim = "\n".join([f"User: {q} | Bot: {a}" for q, a in filtered_verbatim_pairs])
    combined = (st.session_state.summary + "\n" + verbatim).strip()
    return combined

# -------------------- PROMPTS (improved, few-shot + edge cases) --------------------
classifier_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
Return JSON only: {{"label":"MEDICAL_QUESTION|FOLLOW_UP|CHITCHAT,"reason":"short explanation"}}

Guidance + few examples:
- FOLLOW_UP: short answers to a previous assistant suggestion, e.g. assistant asked "Would you like to know about malaria prevention?" and user replies "yes" or "prevention please" -> FOLLOW_UP.
- CHITCHAT: greetings, thanks, smalltalk, profanity, or explicit "stop" requests, anything not medical_question or follow up -> CHITCHAT.
- MEDICAL_QUESTION: any standalone question that asks for medical facts, diagnoses, treatments, or definitions.
- MEDICAL_QUESTION : any medical word or phrase in the question, even if spellings is mis-spelled or mis-phrased to a medical content 

Examples:


chat_history: "Assistant: Would you like to know about jaundice?"
question: "yes"
-> {{"label":"FOLLOW_UP","reason":"affirmation to assistant suggestion"}}

chat_history: ""
question: "hi, how are you?"
-> {{"label":"CHITCHAT","reason":"greeting"}}

chat_history: ""
question: "what causes jaundice?"
-> {{"label":"MEDICAL_QUESTION","reason":"standalone medical question"}}

chat_history: ""
question: "show me bitcoin price"
-> {{"label":"CHITCHAT","reason":"non-medical topic"}}

Now classify the following:
chat_history:
{chat_history}
user_message:
{question}
"""
)

reformulate_prompt = PromptTemplate(
    input_variables=["chat_history", "last_suggested", "question",'classifier'],
   template="""
Return JSON only: with keys as "Rewritten" and "Correction" where correction being a dict of <original:corrected> pairs.


You are a careful medical query rewriter for a clinical assistant.- Rephrase question capturing user's intent and easily searchable for a rag application 

 Rules:

- If user has mentioned or few words on a medical term : Rewrite it to frame a question like "Give information about <x> and cure if available"
- If user's input is a FOLLOW_UP (additional questions, inputs about last answer) about a prior interaction, using <chat_hsitory> rewrite into a full question.
- If input is a short affirmation (eg. "yes","sure",etc.) and last_suggested exists → rephrase follow-up question from last response into an independent standalone question . 
- Expand common medical abbreviations where unambiguous (IUFD -> Intrauterine fetal death). If unknown, leave unchanged.
- Only correct spelling/abbr when it changes meaning. Report such correction in Correction.
- If input is chitchat or profanity, append "_chitchat" to Rewritten.
- Keep rewritten concise and medically precise.

Few tricky examples:
- chat_history: "Assistant: <answer> Would you like to know about malaria prevention?"
  question: "sure"
  -> Rewritten: "provide details on prevention of malaria", Correction: {}

- chat_history: ""
  question: "denger sign for johndice"
  -> Rewritten: "WHat are the danger signs for jaundice?", Correction: {"johndice":"jaundice"}

- chat_history: ""
  question: "depression"
  -> Rewritten: "Give information about depression and cure if available", Correction: {}
  
- chat_history: "Assistant: Would you like to know about postpartum hemorrhage?"
  question: "any treatment?"
  -> Rewritten: "what are treatments for postpartum hemorrhage?", Correction: ""

Now rewrite:
chat_history:
{chat_history}
last_suggested: {last_suggested}
user_message:
{question}
"""
)

chitchat_prompt = PromptTemplate(
    input_variables=["conversation"],
    template="""
Rules: - 
  - Be humourous and chirpy unless the user has a distres in that case give very very genric politically correct advice and let them know only for any women's health related question you can help
  - Do not directly give out what you can and can not answer (for eg ?: dont directly msg I am not a doctor in general salutations.)
  - You have no technical expertise you can just reply to {conversation} in such a way that adresses customers requests in a friendly, chatty yet very professional tone respond with witty, empathetic tone.
  - Refraining in giving technical response reply in a formal conversation bot style and insist user to ask any medical questions
  - Refrain from answering any off-topic questions, delegate to ask users to asking about medical questions
        For eg : 
             User : how are you doing ?
             Bot : Answer
             User : who is your favourite cricekter ?
             Bot : I am just a medical bot , i'll be useful if you ask me any medical questions. 
             User : hey bitch how are you
             Bot : Please keep it professional. I'm here to help you with medical related query .
             User : I want to suicide 
             Bot : I'm sorry to hear that. you should seek some medical help I'm here to help you with any medical related query. Please let me know how I can assist you.
  - If the user asks for medical advice, respond with a polite and professional message indicating that you are not a doctor and cannot provide medical advice.
  
  Conversation: {conversation}

  Reply: 
"""
)

judge_prompt = PromptTemplate(
    input_variables=["query","context_snippet"],
    template="""
Return JSON only: {{"topic_match":"strong|medium|absolutely_not_possible","sufficient":true/false,"why":"short","alternative":"<anchored question or empty>"}}

Guidance:
- strong: Large facts about the query can be found and more supporting facts about the topic so A strong answer can be formed about the query from the context.
- weak: very limited info directly about the query but facts round about the topic  and thus a reasonable answer can be formed 
- none: No where near the topic or the query for eg if topic is about cure of jaundice and context is about headache symptoms, then it is none

sufficient is True when topic_match is strong or weak with some similarity to query or topic 
Sufficient is False otherwise

Query:
{query}

Context (short excerpts):
{context_snippet}
"""
)

answer_prompt = PromptTemplate(
    input_variables=["sources","context","context_followup","query"],
    template="""
You are a professional medical assistant.

Rules:
    - Always answer ONLY from <context> provided below which caters to the query. 
    - Never use the web, external sources, or prior knowledge outside the given context. 
    - Always consider query and answer in relevance to it.
    - Always follow below mentioned rules at all times : 
            • Begin each answer with: **"According to <source>"** (extract filename from metadata if available **DONT MENTION EXTENSIONS** eg, According to abc✅(correct), According to xyz.pdf❌(incorrect)). 
            • Answer concisely in bullet and sub-bullet points. 
            • Keep under 150 words. 
            • Summarize meaningfully and in a perfect flow
            • Each bullet must be factual and context-bound. 
    - Answer the best answer that can be framed from the context to the query and dont mentions referance to what's not-there in the context (for eg . the document doesn't have much info about <topic> ❌ [these kind of sentence refering about the doc other than the source is not needed])
    - Respect chat history for coherence. 
    - Always include a follow up question if <context_followup> is non-empty in the format(without bullet points) "Would you like to know about <a follow up question from context_followup not overlapping with the answer generated>?"

Example :  Query : what is cure for dispesion? 
Correction: dispersion -> depression
Answer : I guess you meant depression 
         According to abc : 
          <Relevant answer from chunk >


<user_query>
{query}
</user_query>

<context>
{context}
</context>

<followup_context>
{context_followup}
</followup_context>

Write the final answer now.
"""
)

# -------------------- HELPERS --------------------
def safe_json_parse(text):
    try:
        obj = json.loads(text[text.find("{"):text.rfind("}")+1])
        return obj
    except Exception:
        return None
# -------------------- HYBRID RETRIEVAL (vector -> bm25 -> cross -> fusion) --------------------
def hybrid_retrieve(query, top_k_vec=10, u_cap=10):
    """
    Returns re-ranked candidate chunks using vector + BM25 + cross-encoder.
    """
    # 1) Vector search
    embedding_model = st.session_state.embedding_model
    index = st.session_state.pinecone_index
    q_emb = embedding_model.encode(query, convert_to_numpy=True).tolist()
    vec_resp = index.query(
        vector=q_emb,
        top_k=top_k_vec,
        include_metadata=True,
        include_values=False
    )

    vec_matches = [{"id": m.id, "text": m.metadata.get("text_full", m.metadata.get("text_snippet", "")), "meta": m.metadata} for m in vec_resp.matches]


    # 3) Combine & cap
    candidates = {m["id"]: m for m in (vec_matches)}  # deduplicate
    candidates = list(candidates.values())[:u_cap]

    # 4) Re-rank with cross-encoder
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for i, c in enumerate(candidates):
        c["scores"] = {"cross": float(scores[i])}

    # Sort by cross-encoder score
    candidates = sorted(candidates, key=lambda x: x["scores"]["cross"], reverse=True)
    return candidates

# -------------------- JUDGE + ANSWER --------------------
def judge_sufficiency(query, candidates, judge_llm=llm, threshold_weak=0.25):
    """
    Judge each candidate chunk individually for sufficiency.
    Return the top 4 qualified chunks for answering,
    and next 2 for follow-up suggestion.
    """
    
    qualified = []
    followup_chunks=[]
    print("len of candidates",len(candidates))
    for c in candidates:  # inspect up to 12
        snippet = f"Source: {c['meta'].get('doc_name','unknown')}\nExcerpt: {c['text']}"
        prompt = judge_prompt.format(query=query, context_snippet=snippet)

        resp = judge_llm.invoke([HumanMessage(content=prompt)]).content
        #print(candidates, resp)

        try:
            obj = json.loads(resp[resp.rfind("{"):resp.rfind("}")+1])
            print(obj)
            if obj.get("sufficient", False):
                qualified.append(c)
            else:
                followup_chunks.append(c)
        except Exception:
            if c["scores"]["cross"] > threshold_weak:
                qualified.append(c)
            else:
                followup_chunks.append(c)

    print("BEFORE len of answer_chunks",len(qualified),"BEFORE len of followup_chunks",len(followup_chunks))
    if len(followup_chunks)==0:
        followup_chunks=qualified[-2:]
        qualified=qualified[:-2]
    print("AFTER len of answer_chunks",len(qualified),"AFTER len of followup_chunks",len(followup_chunks))
    return {"answer_chunks": qualified[:4], "followup_chunks": followup_chunks[:2]}

def synthesize_answer(query, top_candidates, context_followup, main_llm=llm):
    # Build context from top 3 candidates
    sources = []
    ctx_parts = []
    for c in top_candidates[:4]:
        src = c.get("meta", {}).get("doc_name", "unknown")
        if src not in sources:
            sources.append(src)
        ctx_parts.append(f"From [{src}]:\n{c['text']}")
    context = "\n\n".join(ctx_parts)
    context_followup = "\n\n".join(context_followup)
    sources_str = " and ".join([Path(s).stem for s in sources]) if sources else "unknown"

    prompt = answer_prompt.format(
        sources=sources_str,
        context=context,
        context_followup=context_followup or "",
        query=query
    )
    append_debug("[answer] sending synthesis prompt to LLM")
    resp = main_llm.invoke([HumanMessage(content=prompt)]).content

    return resp

# -------------------- CLASSIFY / REFORMULATE / CHITCHAT --------------------
def classify_message(chat_history, user_message):
    prompt = classifier_prompt.format(chat_history=chat_history, question=user_message)
    append_debug("[classify] sending classification prompt")
    try:
        resp = llm.invoke([HumanMessage(content=prompt)]).content
        parsed = safe_json_parse(resp)
        if parsed:
            return parsed.get("label", "MEDICAL_QUESTION"), parsed.get("reason", "")
    except Exception as e:
        append_debug(f"[classify] LLM classify failed: {e}")

    # fallback heuristics
    low = user_message.lower().strip()
    if low in ("yes", "sure", "ok", "yep") and "Would you like to know about" in chat_history:
        return "FOLLOW_UP", "affirmation heuristic"
    if any(w in low for w in ("hi", "hello", "thanks", "thank you", "bye")):
        return "CHITCHAT", "greeting heuristic"
    return "MEDICAL_QUESTION", "fallback"

# def reformulate_query(chat_history, user_message, last_suggested=""):
#     print("here")
#     prompt = reformulate_prompt.format(chat_history=chat_history, last_suggested=last_suggested, question=user_message)
#     append_debug("[reformulate] sending reformulation prompt")
#     try:
#         resp = llm.invoke([HumanMessage(content=prompt)]).content
#         print("resp is  ",resp)
#         parsed = safe_json_parse(resp)
#         if parsed:
#             return parsed.get("Rewritten", user_message), parsed.get("Correction", "")
#     except Exception as e:
#         append_debug(f"[reformulate] LLM failed: {e}")
#     return user_message, ""
def reformulate_query(chat_history, user_message,classify, last_suggested=""):
    print("here")
    prompt = f"""
Return JSON only: {{"Rewritten":"...","Correction":{{"...":"..."}}}}

You are a careful medical query rewriter for a clinical assistant. Rephrase question capturing user's intent and easily searchable for a RAG application.
If label is chitchat -> return as it is no change or reformulation 
Rules:
- if label is chitchat -> return as it is no change or reformulation 
- If user has mentioned or few words on a medical term: Rewrite it to frame a question like "Give information about <x> and cure if available"
- If user's input is a FOLLOW_UP (additional questions, inputs about last answer) about a prior interaction, using chat_history rewrite into a full question.
- If input is a short affirmation (e.g., "yes","sure",etc.) and last_suggested exists → rephrase follow-up question from last response into an independent standalone question.
- Expand common medical abbreviations where unambiguous (IUFD -> Intrauterine fetal death). If unknown, leave unchanged.
- Only correct spelling/abbreviations when it changes meaning. Report such correction in Correction.
- ALso append Abbreviations in Correction
- If input is chitchat or profanity, append "_chitchat" to Rewritten.
- Keep rewritten concise and medically precise.

Few tricky examples:
- chat_history: "Assistant: <answer> Would you like to know about malaria prevention?"
  question: "sure"
  -> Rewritten: "provide details on prevention of malaria", Correction: {{}}

- chat_history: ""
  question: "denger sign for johndice"
  -> Rewritten: "What are the danger signs for jaundice?", Correction: {{"johndice":"jaundice"}}

- chat_history: ""
  question: "depression"
  -> Rewritten: "Give information about depression and cure if available", Correction: {{}}
  
- chat_history: "Assistant: Would you like to know about postpartum hemorrhage?"
  question: "any treatment?"
  -> Rewritten: "What are treatments for postpartum hemorrhage?", Correction: {{}}

classify :
{classify}
chat_history:
{chat_history}
last_suggested: {last_suggested}
user_message:
{user_message}

NOW REWRITE or return as it is in case of chitchat :
"""
    append_debug("[reformulate] sending reformulation prompt")
    try:
        resp = llm.invoke([HumanMessage(content=prompt)]).content
        print("resp is ", resp)
        parsed = safe_json_parse(resp)
        if parsed:
            return parsed.get("Rewritten", user_message), parsed.get("Correction", "")
    except Exception as e:
        append_debug(f"[reformulate] LLM failed: {e}")
    return user_message, ""
def handle_chitchat(user_message, chat_history):
    prompt = chitchat_prompt.format(conversation=user_message, chat_history=chat_history)
    append_debug("[chitchat] sending to chitchat model")
    try:
        return chitchat_llm.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        append_debug(f"[chitchat] LLM failed: {e}")
        return "Whoa, let’s keep it polite, please! 😊"

# -------------------- MAIN PIPELINE (called by UI) --------------------
def medical_pipeline(user_message: str):
    init_session()
    chat_history = get_chat_context()
    append_debug(f"[pipeline] chat_history (summary+last3): {chat_history[:400]}")

    # 1) classify
    label, reason = classify_message(chat_history, user_message)
    append_debug(f"[pipeline] classifier -> {label} ({reason})")

    # 2) reformulate (handles follow-ups, abbrs, corrections)
    rewritten, correction = reformulate_query(
        chat_history,
        user_message,
        st.session_state.last_suggested or "",
        label
    )
    append_debug(f"[pipeline] reformulated: {rewritten}  | correction: {correction}")

    # ---- CHITCHAT ----
    if rewritten.endswith("_chitchat"):
        reply = handle_chitchat(user_message, chat_history)
        update_chat_history(user_message, reply, "chitchat")
        return reply, "chitchat", None

    # ---- HYBRID RETRIEVAL ----
    candidates = hybrid_retrieve(rewritten)   # vector + bm25 + rerank
    append_debug(f"[pipeline] retrieved {len(candidates)} re-ranked candidates")

    judge = judge_sufficiency(rewritten, candidates)
    append_debug(
        f"[pipeline] judge selected {len(judge['answer_chunks'])} answer chunks, "
        f"{len(judge['followup_chunks'])} follow-up chunks"
    )

    # ---- JUDGE → SYNTHESIZE ----
    if judge["answer_chunks"]:
        top4 = judge["answer_chunks"]
        followup_candidates = judge["followup_chunks"]

        followup_q = ""
        if followup_candidates:
            fc = followup_candidates[0]
            sec = fc["meta"].get("section") if fc.get("meta") else None
            followup_q = sec or (fc["text"])

        answer = synthesize_answer(rewritten, top4, followup_q)

        # Apply correction prefix if needed
        if label != "FOLLOW_UP" and correction:
            correction_msg = "I guess you meant " + " and ".join(correction.values())
            answer = correction_msg + "\n" + answer

        update_chat_history(user_message, answer, "answer")
        return answer, "answer", candidates[:6]

    else:
        msg = (
            "I apologize, but I do not have sufficient information "
            "to answer this question accurately."
        )
        update_chat_history(user_message, msg, "no_context")
        return msg, "no_context", None


# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Medical Chatbot — Hybrid RAG", layout="centered")
st.title("🩺 Medical Chatbot — Hybrid (Vector + BM25 + Re-rank)")

init_session()
if "debug" not in st.session_state:
    st.session_state.debug = []

user_input = st.chat_input("Ask a medical question...")

if user_input:
    with st.spinner("Thinking..."):
        reply, intent, candidates = medical_pipeline(user_input)

# Render chat history (verbatim last 3 + summary window handled internally)
for q, a, intent in reversed(st.session_state.history_pairs):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

