# app.py
import os
import json
import time
import math
import hashlib
import logging
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
from rank_bm25 import BM25Okapi
import numpy as np

# LLM / embeddings / vector DB
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Use whatever LLM client you previously used (langchain/openai wrapper).
# Here we provide a minimal wrapper around your `llm` that supports .invoke(prompt) returning .content
# Replace `LLMClient` implementation with your real client class.
from langchain_openai import ChatOpenAI    # if you have this (your earlier code used it)
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# -------------------- Config & init --------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s • %(levelname)s • %(message)s")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY_NEW")

if not OPENAI_API_KEY or not PINECONE_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY or PINECONE_API_KEY_NEW in environment")

# Embedding model (local) — used for BM25 fallback and any local embedding if needed
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = embedding_model.get_sentence_embedding_dimension()

# Pinecone client (new SDK)
pc = Pinecone(api_key=PINECONE_KEY)
INDEX_NAME = os.getenv("PINECONE_INDEX", "medical-chatbot-index")
if INDEX_NAME not in pc.list_indexes().names():
    # if not present, create it (dimension must match embedding dim)
    pc.create_index(name=INDEX_NAME, metric="cosine", dimension=EMBED_DIM, spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(INDEX_NAME)
logging.info("Pinecone index ready.")

# LLM for classification/reformulation/judge/answering
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY, default_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"})

# Local chunks file (maintain during ingestion). If missing, helper below will fetch from Pinecone once.
CHUNKS_STORE_PATH = Path("chunks_store.jsonl")

# -------------------- Helper: load local chunks for BM25 --------------------
def load_chunks_local(path=CHUNKS_STORE_PATH):
    if not path.exists():
        logging.warning("Local chunks file not found. Attempting to export limited metadata from Pinecone (this may be slow).")
        export_chunks_from_pinecone(path)
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    logging.info("Loaded %d local chunks", len(docs))
    return docs

def export_chunks_from_pinecone(path_out, namespace=None, limit_per_fetch=1000):
    """
    Fetch metadata + text_full from Pinecone and write to local JSONL file.
    This requires that ingestion stored full chunk text in metadata as 'text_full' (or 'text').
    """
    logging.info("Exporting chunks from Pinecone into %s ...", path_out)
    # Use index.query with filter None and top_k large? Pinecone doesn't offer full scan easily; here we rely on list_vectors (may be limited).
    # New Pinecone SDK provides index.fetch? We'll use pc.Index().fetch with ids list if you maintain ids elsewhere.
    # As fallback, warn user to create chunks_store.jsonl at ingestion time.
    raise RuntimeError("Local chunks file not found. Please produce `chunks_store.jsonl` at ingestion time containing each chunk as JSON with keys: id, text_full, metadata")

# -------------------- Build BM25 index --------------------
def build_bm25(docs):
    # docs: list of dicts with 'text_full' or 'text'
    tokenized = [d.get("text_full", d.get("text", "")).split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25

# -------------------- Utility: normalize metadata for Pinecone filters --------------------
def clean_meta_for_pinecone(meta: dict):
    safe = {}
    for k, v in (meta or {}).items():
        if v is None:
            safe[k] = "unknown"
        elif isinstance(v, (str, int, float, bool)):
            safe[k] = v
        elif isinstance(v, list):
            safe[k] = [str(x) for x in v if x is not None]
        else:
            safe[k] = str(v)
    return safe

# -------------------- Query classifier prompt (few-shot) --------------------
classifier_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
You are a classifier that labels the user's message into one of three classes: MEDICAL_QUESTION, FOLLOW_UP, CHITCHAT.

Return JSON only: {{"label":"...", "reason":"short"}}

Rules:
- MEDICAL_QUESTION: a standalone medical question requiring factual/clinical info.
- FOLLOW_UP: short replies (yes, sure, more details) that need chat history to form a medical question.
- CHITCHAT: greetings, gratitude, smalltalk, profanity, non-medical requests, or asking the bot to stop.
Use chat_history to decide if the user reply is a follow-up.
chat_history: {chat_history}
user_message: {question}
"""
)

# -------------------- Reformulation prompt (few-shot, concise) --------------------
reformulate_prompt = PromptTemplate(
    input_variables=["chat_history", "question", "last_suggested"],
    template="""
Return output as JSON only: {{"Rewritten":"...","Correction":""}}

You are a high-quality medical query rewriter. Use chat_history and last_suggested (if any) to rewrite the user's message into a clean, standalone medical question.
Rules:
- Preserve user's intent.
- Expand abbreviations if known (e.g., IUFD -> Intrauterine fetal death).
- If user's message is an affirmation and last_suggested is non-empty, rewrite exactly to last_suggested.
- If user's message is clearly chitchat, append "_chitchat" to Rewritten.
- Provide Correction only if you corrected spelling/abbr expansion; else empty.
chat_history: {chat_history}
last_suggested: {last_suggested}
user_message: {question}
"""
)

# -------------------- Judge prompt --------------------
judge_prompt_template = """
You are a liberal medical relevance judge. Return JSON only:
{{"topic_match":"strong|weak|none","sufficient":true/false,"why":"...","alternative":"<anchored question or ''>"}}

Guidance:
- strong: retrieved context matches core topic.
- weak: related angle/population but still useful.
- none: off-topic or noise.

If none -> alternative must be a short medical question (6-16 words) derived from top of context.
<query>
{query}
</query>
<context>
{context}
</context>
"""

# -------------------- Answer synthesis prompt (strict rules) --------------------
answer_prompt_template = """
You are a highly professional medical assistant.

STRICT RULES:
1) Use ONLY the provided CONTEXT. No external knowledge.
2) Begin answer with: "According to {sources}"
   where {sources} are extracted document base names separated by ' and '.
3) Answer in bullet points (no paragraphs). Max 150 words.
4) If correction exists, put it on the first line (not counted as a bullet).
5) If context_followup non-empty, append single-line follow-up question exactly:
   Would you like to know about <followup question>?

Context:
{context}

Followups:
{context_followup}

User query:
{query}
"""

# -------------------- Scoring fuse function --------------------
def fuse_scores(bm25_score, vector_score, meta_score, bm25_w=0.4, vec_w=0.5, meta_w=0.1):
    # normalize bm25 (already ratio), assume vector_score in [-1,1] cosine -> map to [0,1]
    vs = (vector_score + 1) / 2
    # meta_score already in [0,1]
    fused = bm25_w * bm25_score + vec_w * vs + meta_w * meta_score
    return fused

# -------------------- Retrieval pipeline --------------------
def hybrid_retrieve(query, bm25, local_docs, top_k=10):
    """
    Returns a list of candidate chunks with fused score and metadata.
    local_docs: list of dicts with keys: 'id','text_full','metadata'
    bm25: trained BM25 index on tokenized texts
    """
    # 1) BM25 lexical scores
    tokenized_q = query.split()
    bm25_scores = bm25.get_scores(tokenized_q)  # array aligned to local_docs

    # 2) Vector retrieval from Pinecone
    q_emb = embedding_model.encode(query, convert_to_numpy=True).tolist()
    vec_resp = index.query(q_emb, top_k=top_k, include_metadata=True, include_values=False)
    # vec_resp.matches is list of matches with id, score, metadata
    vec_matches = {m.id: m.score for m in (vec_resp.matches or [])}

    # 3) Build candidate set (union of top BM25 and top vector)
    # get top bm25 indices
    top_bm25_idx = np.argsort(bm25_scores)[-top_k:][::-1]
    candidate_ids = set()
    for idx in top_bm25_idx:
        candidate_ids.add(local_docs[idx]["id"])
    for mid, score in vec_matches.items():
        candidate_ids.add(mid)

    # 4) compute fused score per candidate
    candidates = []
    # Precompute bm25 normalization (divide by max)
    bm_max = float(np.max(bm25_scores)) if len(bm25_scores)>0 else 1.0
    for cid in candidate_ids:
        # find local doc
        doc = next((d for d in local_docs if d["id"] == cid), None)
        # if not in local docs, try fetching metadata from pinecone match list
        if doc is None:
            # attempt to fetch from pinecone metadata
            fetch = index.fetch(ids=[cid], include_metadata=True)
            meta = fetch.vectors.get(cid, {}).get("metadata", {}) if getattr(fetch, "vectors", None) else {}
            text_full = meta.get("text_full", meta.get("text", meta.get("text_snippet", "")))
            doc = {"id": cid, "text_full": text_full, "metadata": meta}
        bm25_raw = 0.0
        if doc in local_docs:
            idx = local_docs.index(doc)
            bm25_raw = bm25_scores[idx]
        bm25_norm = bm25_raw / bm_max if bm_max else 0.0
        vec_score = vec_matches.get(cid, -1.0)
        # metadata score: boost if same section or doc_name match common tokens with query
        meta = doc.get("metadata", {})
        meta_score = 0.0
        if meta.get("section") and meta.get("section").lower() in query.lower():
            meta_score = 1.0
        # fused
        fused = fuse_scores(bm25_norm, vec_score, meta_score)
        candidates.append({"id": cid, "text": doc.get("text_full"), "meta": meta, "scores": {"bm25": bm25_norm, "vec": vec_score, "meta": meta_score, "fused": fused}})
    # sort by fused descending
    candidates = sorted(candidates, key=lambda x: x["scores"]["fused"], reverse=True)
    return candidates[:top_k]

# -------------------- Liberal judge: uses LLM to decide sufficiency --------------------
def judge_sufficiency(query, top_candidates, judge_llm=llm, threshold_weak=0.25):
    """
    top_candidates: list of candidate dicts from hybrid_retrieve
    Returns dict with keys: sufficient(bool), topic_match, alternative (anchored)
    We'll ask the LLM to inspect short excerpts and decide.
    """
    # build a compact context snippet for judge
    cnt = 6
    snippet = "\n\n".join([f"Source: {c['meta'].get('doc_name','unknown')}\nExcerpt: {c['text'][:400]}" for c in top_candidates[:cnt]])
    prompt = judge_prompt_template.format(query=query, context=snippet)
    resp = judge_llm.invoke(HumanMessage(content=prompt)).content
    # extract last JSON
    try:
        # naive: find last {...}
        obj = json.loads(resp[resp.rfind("{"):resp.rfind("}")+1])
    except Exception:
        # fallback: simple heuristic: if fused score sum high enough -> true
        avg_fused = np.mean([c["scores"]["fused"] for c in top_candidates]) if top_candidates else 0.0
        if avg_fused > threshold_weak:
            return {"sufficient": True, "topic_match":"weak", "alternative": ""}
        return {"sufficient": False, "topic_match":"none", "alternative": ""}
    return obj

# -------------------- Answer generator --------------------
def synthesize_answer(query, top_candidates, context_followup, correction, main_llm=llm):
    # Build context string and extract top source names
    sources = []
    ctx_parts = []
    for c in top_candidates[:4]:
        src = c["meta"].get("doc_name", "unknown")
        if src not in sources:
            sources.append(src)
        ctx_parts.append(f"From [{src}]:\n{c['text']}")
    context = "\n\n".join(ctx_parts)
    sources_str = " and ".join([Path(s).stem for s in sources]) if sources else "unknown"
    answer_prompt = answer_prompt_template.format(context=context, context_followup=context_followup or "", query=query)
    # Prepend correction if exists
    if correction:
        answer_prompt = correction + "\n\n" + answer_prompt
    resp = main_llm.invoke(HumanMessage(content=answer_prompt)).content
    # ensure starts with According to ...
    if not resp.strip().lower().startswith("according to"):
        resp = f"According to {sources_str}\n\n{resp}"
    return resp

# -------------------- Reformulate & classify wrappers --------------------
def classify_message(chat_history, user_message):
    prompt = classifier_prompt.format(chat_history=chat_history, question=user_message)
    resp = llm.invoke(HumanMessage(content=prompt)).content
    try:
        obj = json.loads(resp[resp.rfind("{"):resp.rfind("}")+1])
        return obj.get("label","MEDICAL_QUESTION"), obj.get("reason","")
    except Exception:
        # fallback heuristics
        if len(user_message.split()) <= 3 and user_message.lower() in ("yes","sure","ok","no","nope","nah","never mind","nevermind"):
            return "FOLLOW_UP","short reply"
        if any(w in user_message.lower() for w in ["hi","hello","how are you","thanks","thank you","bye"]):
            return "CHITCHAT","greeting"
        return "MEDICAL_QUESTION","fallback"

def reformulate_query(chat_history, user_message, last_suggested=""):
    prompt = reformulate_prompt.format(chat_history=chat_history, question=user_message, last_suggested=last_suggested)
    resp = llm.invoke(HumanMessage(content=prompt)).content
    try:
        obj = json.loads(resp[resp.rfind("{"):resp.rfind("}")+1])
        return obj.get("Rewritten", user_message), obj.get("Correction","")
    except Exception:
        return user_message, ""

# -------------------- Main pipeline --------------------
def medical_pipeline(user_message, chat_history, last_suggested, local_docs, bm25):
    # 1) profanity quick block (basic)
    if contains_profanity(user_message):
        return "Whoa, let’s keep it polite, please! 😊", "chitchat", None

    # 2) classify
    label, reason = classify_message(chat_history, user_message)
    logging.info("Classifier label=%s reason=%s", label, reason)

    if label == "CHITCHAT":
        # build chitchat reply
        reply = llm.invoke(HumanMessage(content=f"You are a cheerful assistant. Short friendly reply to: {user_message}")).content
        return reply, "chitchat", None

    # 3) reformulate
    rewritten, correction = reformulate_query(chat_history, user_message, last_suggested or "")
    logging.info("Rewritten: %s | Correction: %s", rewritten, correction)

    # Special-case: reformulator can mark appended "_chitchat"
    if rewritten.endswith("_chitchat"):
        content = rewritten.replace("_chitchat", "").strip()
        reply = llm.invoke(HumanMessage(content=f"Chitchat response to: {content}")).content
        return reply, "chitchat", None

    # 4) retrieval (hybrid)
    candidates = hybrid_retrieve(rewritten, bm25, local_docs, top_k=12)
    logging.info("Top candidate scores (fused): %s", [round(c['scores']['fused'],3) for c in candidates[:6]])

    # 5) judge sufficiency
    judge = judge_sufficiency(rewritten, candidates)
    logging.info("Judge result: %s", judge)

    if judge.get("sufficient"):
        # choose top 4 to answer
        top4 = candidates[:4]
        # context_followup derive from next 2 candidates (simple)
        followup_candidates = candidates[4:6] or []
        followup_q = ""
        if followup_candidates:
            # build an anchored follow-up using section title or short phrase
            fc = followup_candidates[0]
            sec = fc["meta"].get("section") or fc["meta"].get("type") or ""
            followup_q = sec if sec else (fc["text"][:80])
        answer = synthesize_answer(rewritten, top4, followup_q, correction)
        return answer, "answer", candidates[:6]
    else:
        alt = judge.get("alternative","").strip()
        if alt:
            # store alt for next yes
            st.session_state.last_suggested = alt
            return f"I apologize, but I do not have sufficient information to answer this question accurately. Would you like to know about {alt} instead?", "no_context", None
        return "I apologize, but I do not have sufficient information to answer this question accurately.", "no_context", None

# -------------------- Profanity helper --------------------
BAD_WORDS = ["fuck","shit","bitch","asshole","bastard","slut"]
def contains_profanity(msg):
    m = msg.lower()
    return any(w in m for w in BAD_WORDS)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Medical Chatbot (Hybrid RAG)", layout="centered")
st.title("🩺 Medical Chatbot — Hybrid RAG")

if "history" not in st.session_state:
    st.session_state.history = []   # list of (user, assistant, intent)
if "last_suggested" not in st.session_state:
    st.session_state.last_suggested = ""

# Load local docs & BM25 once
if "local_docs" not in st.session_state:
    local_docs = load_chunks_local()
    st.session_state.local_docs = local_docs
    st.session_state.bm25 = build_bm25(local_docs)
else:
    local_docs = st.session_state.local_docs
    bm25 = st.session_state.bm25

user_input = st.chat_input("Ask a medical question...")

if user_input:
    with st.spinner("Thinking..."):
        # Build chat_history string for LLMs (last 6 messages)
        hist = "\n".join([f"User: {q} | Bot: {a}" for q,a,_ in st.session_state.history[-6:]])
        bm25 = st.session_state.bm25
        answer, intent, candidates = medical_pipeline(user_input, hist, st.session_state.last_suggested, local_docs, bm25)
        st.session_state.history.append((user_input, answer, intent))

# render history
for q,a,intent in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# debugging pane
with st.expander("Debug / Top retrieved (for last query)"):
    cand = None
    if st.session_state.history:
        last_user, last_answer, last_intent = st.session_state.history[-1]
        # Attempt to show candidates if stored in last pipeline run
        # We used a transient store for candidates; if available, show top 6
    st.write("To inspect retrieval, run queries and check logs. Candidate lists (id, fused) are printed in server logs.")
