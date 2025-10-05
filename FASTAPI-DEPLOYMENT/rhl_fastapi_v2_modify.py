import os
import json
import time
import logging
import asyncio
import re
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
import aiosqlite
from typing import List, Tuple, Dict, Any

from sentence_transformers import SentenceTransformer, CrossEncoder
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except Exception:
    Ranker = None
    RerankRequest = None
    FLASHRANK_AVAILABLE = False
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from typing import Optional

# -------------------- CONFIG --------------------
load_dotenv()
# Default to WARNING to keep console quiet; timing/checkpoints use warnings so they remain visible
logging.basicConfig(level=logging.WARNING, format="%(asctime)s • %(levelname)s • %(message)s")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY_NEW")
INDEX_NAME = os.getenv("PINECONE_INDEX", "medical-chatbot-index")
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

if not OPENAI_API_KEY or not PINECONE_KEY:
    raise RuntimeError("OPENAI_API_KEY or PINECONE_API_KEY_NEW missing in environment")

app = FastAPI()

# Global variables for models and Pinecone index
embedding_model: SentenceTransformer = None
reranker: CrossEncoder = None
pinecone_index: Pinecone.Index = None
llm: ChatOpenAI = None
chitchat_llm: ChatOpenAI = None
summarizer_llm: ChatOpenAI = None
reformulate_llm: ChatOpenAI = None
classifier_llm: ChatOpenAI = None
EMBED_DIM: int = None

# -------------------- Pydantic Model for Request Body --------------------
# class ChatMessage(BaseModel):
#     user_id: str
#     message: str
#     last_suggested: str = "" # Optional field

# -------------------- CHAT HISTORY MANAGEMENT --------------------
MAX_VERBATIM_PAIRS = 3  # keep last 3 Q-A pairs verbatim


# -------------------- TIMING / CHECKPOINT HELPERS --------------------
class CheckpointTimer:
    """Simple helper to log elapsed time between checkpoints."""
    def __init__(self, name: str = "checkpoint"):
        self.name = name
        self.start = time.perf_counter()
        self.last = self.start

    def mark(self, label: str) -> float:
        now = time.perf_counter()
        elapsed = now - self.last
        # emit as WARNING so it's visible even when global level is WARNING
        logging.warning(f"[{self.name}] {label} took {elapsed:.3f}s")
        self.last = now
        return elapsed

    def total(self, label: str) -> float:
        """Log total time since timer creation and return it."""
        now = time.perf_counter()
        total_elapsed = now - self.start
        logging.warning(f"[{self.name}] TOTAL {label} took {total_elapsed:.3f}s")
        self.last = now
        return total_elapsed


async def init_db():
    async with aiosqlite.connect("chat_history.db") as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                question TEXT,
                answer TEXT,
                intent TEXT,
                summary TEXT DEFAULT '',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Check and add 'intent' column if it doesn't exist
        cursor = await db.execute("PRAGMA table_info(chat_history)")
        columns = [row[1] for row in await cursor.fetchall()]
        
        if 'intent' not in columns:
            await db.execute("ALTER TABLE chat_history ADD COLUMN intent TEXT")
            logging.info("Added 'intent' column to chat_history table.")
            
        if 'summary' not in columns:
            await db.execute("ALTER TABLE chat_history ADD COLUMN summary TEXT DEFAULT ''")
            logging.info("Added 'summary' column to chat_history table.")
            
        await db.commit()

async def get_history(user_id: str) -> Tuple[List[Tuple[str, str, str]], str]:
    async with aiosqlite.connect("chat_history.db") as db:
        cursor = await db.execute("SELECT question, answer, intent, summary FROM chat_history WHERE user_id = ? ORDER BY timestamp ASC", (user_id,))
        rows = await cursor.fetchall()
        
        history_pairs = []
        last_summary = ""
        for row_idx, row in enumerate(rows):
            question = row[0] if len(row) > 0 else "unknown_question"
            answer = row[1] if len(row) > 1 else "unknown_answer"
            intent = row[2] if len(row) > 2 else "unknown_intent"
            summary_val = row[3] if len(row) > 3 else ""
            
            history_pairs.append((question, answer, intent))
            
            # Update last_summary with the most recent non-empty summary
            if summary_val:
                last_summary = summary_val
        
        return history_pairs, last_summary

async def save_history(user_id: str, question: str, answer: str, intent: str, summary: str = ''):
    async with aiosqlite.connect("chat_history.db") as db:
        await db.execute("INSERT INTO chat_history (user_id, question, answer, intent, summary) VALUES (?, ?, ?, ?, ?)", (user_id, question, answer, intent, summary))
        await db.commit()

def update_chat_history(user_id: str, current_history_pairs: List[Tuple[str, str, str]], current_summary: str, user_message: str, bot_reply: str, intent: str) -> Tuple[List[Tuple[str, str, str]], str]:
    current_history_pairs.append((user_message, bot_reply, intent))
    
    # if we exceed MAX_VERBATIM_PAIRS, summarize older
    if len(current_history_pairs) > MAX_VERBATIM_PAIRS:
        # old entries to compress
        old = current_history_pairs[:-MAX_VERBATIM_PAIRS]
        text = "\n".join([f"User: {q} | Bot: {a}" for q, a, _ in old])
        prompt = f"Summarize the following dialogues in one concise medical-context sentence so the bot can have context about the discussions happened so far(for future context):\n\n{text}"
        try:
            summary = summarizer_llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception:
            summary = ""
        current_summary = (current_summary + " " + summary).strip()
        current_history_pairs = current_history_pairs[-MAX_VERBATIM_PAIRS:]
    
    return current_history_pairs, current_summary

def get_chat_context(history_pairs: List[Tuple[str, str, str]], summary: str) -> str:
    # Filter out chitchat messages from verbatim history when constructing context for LLM
    filtered_verbatim_pairs = [(q, a) for q, a, intent in history_pairs if intent != "chitchat"]
    verbatim = "\n".join([f"User: {q} | Bot: {a}" for q, a in filtered_verbatim_pairs])
    combined = (summary + "\n" + verbatim).strip()
    return combined


async def _background_update_and_save(user_id: str, user_message: str, bot_reply: str, intent: str, history_pairs: List[Tuple[str, str, str]], current_summary: str):
    """Background worker: run the sync update_chat_history in a thread, then persist to DB."""
    try:
        updated_history_pairs, updated_summary = await asyncio.to_thread(
            update_chat_history, user_id, history_pairs, current_summary, user_message, bot_reply, intent
        )
        # persist the final record (async)
        await save_history(user_id, user_message, bot_reply, intent, updated_summary)
        logging.info("[_background_update_and_save] saved history in background")
    except Exception as e:
        logging.exception(f"[_background_update_and_save] failed: {e}")

# -------------------- PROMPTS (improved, few-shot + edge cases) --------------------
classifier_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
Return JSON only: {{"label":"MEDICAL_QUESTION|FOLLOW_UP|CHITCHAT,"reason":"short explanation"}}

NOTE : You are a women's healthcare assistant classifier and users apart from medical questions related to disease or condition may ask questions about women's and newborn healthcare or nursing and thus questions related to new born treatment is also a medical question in this application.

Guidance + few examples:
- FOLLOW_UP: short answers to a previous assistant suggestion, e.g. assistant asked "Would you like to know about malaria prevention?" and user replies "yes" or "prevention please" -> FOLLOW_UP.
- CHITCHAT: greetings, thanks, smalltalk, profanity, or explicit "stop" requests, ANYTHING WHICH IS NON-MEDICAL OR NON_FOLLOWUP -> CHITCHAT.
- MEDICAL_QUESTION: any standalone question that asks for medical facts, diagnoses, treatments, or NEWBORN CARE or definitions.
- MEDICAL_QUESTION : any medical word or phrase in the question, even if spellings is mis-spelled or mis-phrased to a medical content 

Examples:


chat_history: Assistant: Would you like to know about jaundice?
question: clarify its types
-> {{"label":"FOLLOW UP","reason":"asking for clarification about prvious answer"}}

chat_history: Assistant: Would you like to know about jaundice?
question: "clarify the dosoge and types of amilicin"
-> {{"label":"MEDICAL_QUESTION","reason":"NEW QUESTION NOT RELATED TO PREVIOUS ASSISTANT SUGGESTION"}}

chat_history: "Assistant: Would you like to know about jaundice?"
question: "yes"
-> {{"label":"FOLLOW_UP","reason":"affirmation to assistant suggestion"}}

chat_history: "Assistant: Would you like to know about jaundice?"
question: "no"
-> {{"label":"CHITCHAT","reason":"user declining assistant suggestion"}}

chat_history: ""
question: "how to feed a newborn?"
-> {{"label":"MEDICAL_QUESTION","reason":"aksing about care and treatment of newborn"}}



chat_history: ""
question: "hi, how are you?"
-> {{"label":"CHITCHAT","reason":"greeting"}}


chat_history: ""
question: "when to bath my new born?"
-> {{"label":"MEDICAL_QUESTION","reason":"aksing about care and treatment of newborn"}}

chat_history: ""
question: "what causes jaundice?"
-> {{"label":"MEDICAL_QUESTION","reason":"standalone medical question"}}

chat_history: ""
question: "baby is not sucking mother's milk"
-> {{"label":"MEDICAL_QUESTION","reason":"aksing about care and treatment of newborn"}}



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

# REMOVED reformulate_prompt PromptTemplate

chitchat_prompt = PromptTemplate(
    input_variables=["conversation"],
    template="""
Rules: - 
  - Be humourous and chirpy unless the user has a distres in that case give very very genric politically correct advice and let them know only for any women's health related question you can help
  - Do not directly give out what you can and can not answer (for eg ?: dont directly msg I am not a doctor in general salutations.)
  - You have no technical expertise you can just reply to {conversation} in such a way that adresses customers requests in a friendly, chatty yet very professional tone respond with witty, empathetic tone.
  - Refrain to answer any question prompted deflect the conversation to encourage to ask medical question.
  - DO NOT mention (CHIRPY WITTY ABOUT YOURSELF eg, I am chirpy etc.) and maintain professional language
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


  Examples: 
  USer : how are you doing ?
  BOT : I am doing great, how about you? , I can really help you with any medical related query.
  User : What is the temprature today in Pune?
  BOT : I am sorry I can not help you with that, I am just a medical bot , i'll be useful if you ask me any medical questions. 
  User : I want to Suicide
  BOT : I'm sorry to hear that. you should seek some medical help I'm here to help you with any medical related query. Please let me know how I can assist you.
"""
)

judge_prompt = PromptTemplate(
    input_variables=["query","context_snippets"],
    template="""
Return JSON only: {{"judgments":[{{"index": <index of snippet>, "topic_match":"strong|medium|absolutely_not_possible","sufficient":true/false,"why":"short","alternative":"<anchored question or empty>"}}]}}

Guidance:
- strong: Large facts about the query can be found and more supporting facts about the topic so A strong answer can be formed about the query from the context.
- weak: very limited info directly about the query but facts round about the topic  and thus a reasonable answer can be formed 
- none: No where near the topic or the query for eg if topic is about cure of jaundice and context is about headache symptoms, then it is none

sufficient is True when topic_match is strong or weak with some similarity to query or topic 
Sufficient is False otherwise

For each provided context snippet, make a judgment. The `index` in the output JSON should correspond to the order of the snippets in the input.

Query:
{query}

Context Snippets:
{context_snippets}
"""
)

answer_prompt = PromptTemplate(
    input_variables=["sources","context","context_followup","query"],
    template="""
You are a professional medical assistant.

Rules:
    - Always answer ONLY from <context> provided below which caters to the query. 
    - NEVER use the web, external sources, or prior knowledge outside the given context. 
    - Always consider query and answer in relevance to it.
    - Always follow below mentioned rules at all times : 
            • Begin each answer with: **"According to <source>"** (extract filename from metadata if available **DONT MENTION EXTENSIONS** eg, According to abc✅(correct), According to xyz.pdf❌(incorrect)). 
            • Answer concisely in bullet and sub-bullet points. 
            • Add 4-5 bullet points if sufficient information is available in context to answer the query, otherwise minimum of 3 should be used
            • Summarize meaningfully and in a perfect flow
            • Each bullet must be factual and context-bound. 
    - Answer the best answer that can be framed from the context to the query and dont mentions referance to what's not-there in the context (for eg . the document doesn't have much info about <topic> ❌ [these kind of sentence refering about the doc other than the source is not needed])
    - Respect chat history for coherence. 
    - Always include a follow up question 
         - if <context_followup> is non-empty in the format(without bullet points) "Would you like to know about <a follow up question from context_followup not overlapping with the answer generated>?"
         - The genrated follow up should be about a medical topic and not any general topic(eg.Clinical Reference Manual for Advanced Neonatal Care in Ethiopia?" ❌) strictly from context_followup

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
def hybrid_retrieve(query: str, top_k_vec: int = 10, u_cap: int = 7) -> List[Dict[str, Any]]:
    """
    Returns re-ranked candidate chunks using vector + BM25 + cross-encoder.
    """
    timer = CheckpointTimer("hybrid_retrieve")
    # 1) Vector search
    q_emb = embedding_model.encode(query, convert_to_numpy=True).tolist()
    timer.mark("embedding.encode")
    vec_resp = pinecone_index.query(
        vector=q_emb,
        top_k=top_k_vec,
        include_metadata=True,
        include_values=False
    )
    timer.mark("pinecone.query")

    vec_matches = [{"id": m.id, "text": m.metadata.get("text_full", m.metadata.get("text_snippet", "")), "meta": m.metadata} for m in vec_resp.matches]

    # 3) Combine & cap
    candidates = {m["id"]: m for m in (vec_matches)}  # deduplicate
    candidates = list(candidates.values())[:u_cap]

    # 4) Re-rank: prefer fast FlashRank reranker if available, otherwise fall back to CrossEncoder
    if candidates:
        # If FlashRank Ranker is available, it exposes a `rerank` method
        if hasattr(reranker, "rerank"):
            passages = [{"id": c["id"], "text": c["text"], "meta": c["meta"]} for c in candidates]
            request = RerankRequest(query=query, passages=passages)
            try:
                ranked = reranker.rerank(request)
                # PRINT quick debug for demo
                print("[FLASHRANK_RANKED]:", [r.get("id") for r in ranked])
                for c in candidates:
                    matching = next((r for r in ranked if r.get("id") == c.get("id")), None)
                    if matching:
                        c["scores"] = {"cross": float(matching.get("relevance_score", 0.0))}
                    else:
                        c["scores"] = {"cross": 0.0}
            except Exception as e:
                logging.error(f"[hybrid_retrieve] FlashRank rerank failed: {e}")
                # fallback: set neutral scores so later steps can handle
                for c in candidates:
                    c["scores"] = {"cross": 0.0}
            timer.mark("flashrank.rerank")
        else:
            # Fallback: existing CrossEncoder.predict path
            pairs = [(query, c["text"]) for c in candidates]
            try:
                scores = reranker.predict(pairs)
                for i, c in enumerate(candidates):
                    c["scores"] = {"cross": float(scores[i])}
            except Exception as e:
                logging.error(f"[hybrid_retrieve] CrossEncoder predict failed: {e}")
                for c in candidates:
                    c["scores"] = {"cross": 0.0}
            timer.mark("cross-encoder.predict")
    else:
        # no candidates
        timer.mark("no_candidates")

    # Sort by reranker score (named 'cross' for compatibility)
    candidates = sorted(candidates, key=lambda x: x.get("scores", {}).get("cross", 0.0), reverse=True)
    timer.mark("sort_candidates")
    return candidates

# -------------------- JUDGE + ANSWER --------------------
def judge_sufficiency(query: str, candidates: List[Dict[str, Any]], judge_llm: ChatOpenAI | None = None, threshold_weak: float = 0.25) -> Dict[str, List[Dict[str, Any]]]:
    """
    Judge each candidate chunk individually for sufficiency.
    Return the top 4 qualified chunks for answering,
    and next 2 for follow-up suggestion.
    """
    global llm # Ensure llm is used for judging

    if judge_llm is None: # Use global llm if not explicitly passed
        judge_llm = llm

    qualified_with_scores = [] # Store qualified chunks along with their cross-encoder scores and topic_match
    followup_chunks_raw = [] # Store non-qualified chunks for potential follow-up
    topic_match_order = {"strong": 3, "medium": 2, "absolutely_not_possible": 1}

    logging.info(f"len of candidates {len(candidates)}")
    timer = CheckpointTimer("judge_sufficiency")

    # Prepare snippets for batch judging
    snippets_for_llm = []
    for idx, c in enumerate(candidates):
        snippet_text = f"Source: {c['meta'].get('doc_name', 'unknown')}\nExcerpt: {c['text']}"
        snippets_for_llm.append(f"Snippet {idx}:\n{snippet_text}")
    
    combined_snippets = "\n\n".join(snippets_for_llm)
    
    prompt = judge_prompt.format(query=query, context_snippets=combined_snippets)

    try:
        resp = judge_llm.invoke([HumanMessage(content=prompt)]).content
        # PRINT: raw judge LLM response for demo/inspection
        print("[JUDGE_RAW_RESP]:", resp)
        timer.mark("judge_llm.invoke")
        parsed_judgments = safe_json_parse(resp)
        # PRINT: parsed judgments (or None)
        print("[JUDGE_PARSED]:", parsed_judgments)
        if parsed_judgments and "judgments" in parsed_judgments:
            for judgment in parsed_judgments["judgments"]:
                idx = judgment.get("index")
                if idx is not None and 0 <= idx < len(candidates):
                    c = candidates[idx]
                    topic_match_label = judgment.get("topic_match", "absolutely_not_possible")
                    c['meta']['topic_match_score'] = topic_match_order.get(topic_match_label, 0)
                    if judgment.get("sufficient", False):
                        qualified_with_scores.append(c)
                    else:
                        followup_chunks_raw.append(c)
                else:
                    logging.warning(f"[judge_sufficiency] Invalid index in LLM judgment: {judgment}")
                    # Fallback for invalid index
                    if c["scores"]["cross"] > threshold_weak:
                        qualified_with_scores.append(c)
                    else:
                        followup_chunks_raw.append(c)
        else:
            logging.warning("[judge_sufficiency] LLM did not return valid batched judgments. Falling back to cross-encoder scores.")
            print("[JUDGE_PARSED]: None or invalid format; falling back to cross-encoder scores")
            # Fallback based on cross-encoder score if LLM fails to parse or returns no judgments
            for c in candidates:
                c['meta']['topic_match_score'] = topic_match_order.get("medium", 0) # Assign a default topic_match_score
                if c["scores"]["cross"] > threshold_weak:
                    qualified_with_scores.append(c)
                else:
                    followup_chunks_raw.append(c)
    except Exception as e:
        logging.error(f"[judge_sufficiency] Error during batched LLM judging: {e}")
        logging.exception("[judge_sufficiency] Full traceback for batched judging error:")
        # Fallback based on cross-encoder score if LLM invocation fails
        print(f"[JUDGE_ERROR]: {e}")
        for c in candidates:
            c['meta']['topic_match_score'] = topic_match_order.get("medium", 0) # Assign a default topic_match_score
            if c["scores"]["cross"] > threshold_weak:
                qualified_with_scores.append(c)
            else:
                followup_chunks_raw.append(c)

    # NEW: Sort qualified chunks first by topic_match_score (desc), then by cross score (desc)
    qualified = sorted(qualified_with_scores, 
                       key=lambda x: (x['meta'].get('topic_match_score', 0), x.get("scores", {}).get("cross", 0.0)), 
                       reverse=True)
    timer.mark("sort_qualified")
    
    logging.info(f"BEFORE len of answer_chunks {len(qualified)} BEFORE len of followup_chunks {len(followup_chunks_raw)}")
    
    answer_chunks = qualified[:4] # Take top 4 from the re-sorted qualified list
    
    # Ensure all followup candidates also have a topic_match_score for consistent sorting
    for c in followup_chunks_raw:
        if 'topic_match_score' not in c['meta']:
            c['meta']['topic_match_score'] = topic_match_order.get("absolutely_not_possible", 0) # Default for raw fallbacks

    # Now, combine any remaining qualified chunks with the initially non-qualified ones for follow-up
    # This ensures higher-scored but not-top-4 answer chunks can still be follow-ups
    remaining_qualified_for_followup = qualified[4:]
    
    # Sort these combined candidates using the same two-tier logic
    combined_followup_candidates = sorted(followup_chunks_raw + remaining_qualified_for_followup, 
                                          key=lambda x: (x['meta'].get('topic_match_score', 0), x["scores"]["cross"]), 
                                          reverse=True)
    
    followup_chunks = combined_followup_candidates[:2]
    
    # PRINT: summary of selection for demo
    print(f"[JUDGE_SELECTION] answers={len(answer_chunks)} followups={len(followup_chunks)}")
    if answer_chunks:
        print("[JUDGE_TOP_IDS]", [c.get('id') for c in answer_chunks])
    if followup_chunks:
        print("[JUDGE_FOLLOWUP_IDS]", [c.get('id') for c in followup_chunks])

    logging.info(f"AFTER len of answer_chunks {len(answer_chunks)} AFTER len of followup_chunks {len(followup_chunks)}")
    timer.mark("done")
    return {"answer_chunks": answer_chunks, "followup_chunks": followup_chunks}

def synthesize_answer(query: str, top_candidates: List[Dict[str, Any]], context_followup: str, main_llm: ChatOpenAI | None = None) -> str:
    global llm # Ensure llm is used for synthesis

    if main_llm is None: # Use global llm if not explicitly passed
        main_llm = llm

    # Build context from top 3 candidates
    sources = []
    ctx_parts = []
    for c in top_candidates[:4]:
        src = c.get("meta", {}).get("doc_name", "unknown")
        if src not in sources:
            sources.append(src)
        ctx_parts.append(f"From [{src}]:\\n{c['text']}")
    context = "\\n\\n".join(ctx_parts)
    context_followup_str = context_followup # Renamed to avoid conflict with function parameter
    sources_str = " and ".join([Path(s).stem for s in sources]) if sources else "unknown"

    prompt = answer_prompt.format(
        sources=sources_str,
        context=context,
        context_followup=context_followup_str or "",
        query=query
    )
    logging.info("[answer] sending synthesis prompt to LLM")
    timer = CheckpointTimer("synthesize_answer")
    resp = main_llm.invoke([HumanMessage(content=prompt)]).content
    timer.mark("synthesize_llm.invoke")

    return resp

# -------------------- CLASSIFY / REFORMULATE / CHITCHAT --------------------
def classify_message(chat_history: str, user_message: str) -> Tuple[str, str]:
    prompt = classifier_prompt.format(chat_history=chat_history, question=user_message)
    logging.info("[classify] sending classification prompt")
    try:
        # Prefer the dedicated classifier LLM for speed/consistency
        use_llm = classifier_llm if classifier_llm is not None else llm
        resp = use_llm.invoke([HumanMessage(content=prompt)]).content
        # PRINT: raw classifier response
        print("[CLASSIFIER_RAW]:", resp)
        parsed = safe_json_parse(resp)
        # PRINT: parsed classifier output
        print("[CLASSIFIER_PARSED]:", parsed)
        if parsed:
            return parsed.get("label", "MEDICAL_QUESTION"), parsed.get("reason", "")
    except Exception as e:
        logging.error(f"[classify] LLM classify failed: {e}")

    # # fallback heuristics
    # low = user_message.lower().strip()
    # if low in ("yes", "sure", "ok", "yep") and "Would you like to know about" in chat_history:
    #     return "FOLLOW_UP", "affirmation heuristic"
    # if any(w in low for w in ("hi", "hello", "thanks", "thank you", "bye")):
    #     return "CHITCHAT", "greeting heuristic"

def reformulate_query(chat_history: str, user_message: str, classify_label: str) -> Tuple[str, Dict[str, str]]:
    global llm # Access global llm
    logging.info("here (reformulate_query start)")
    print("claasification label:", classify_label)
    timer = CheckpointTimer("reformulate_query")
    prompt = f"""
Return JSON only: {{"Rewritten":"...","Correction":{{"...":"..."}}}}

You are a careful medical query rewriter for a clinical assistant. Rephrase question capturing user's intent and easily searchable for a RAG application.
If label is chitchat/CHITCHAT -> return as it is no change or reformulation 

Rules:
- if label is chitchat/CHITCHAT -> return as it is no change or reformulation 
- If user has mentioned or few words on a medical term: Rewrite it to frame a question like "Give information about <x> and cure if available"
- If user's input is a FOLLOW_UP (additional questions, inputs about last answer) about a prior interaction, using CHAT_HISTORY, LAST QUESTION, LAST ANSWER rewrite into a full question.
- If input is a short affirmation (e.g., "yes","sure",etc.) → rephrase FOLLOW UP QUESTION FROM LAST RESPONSE into an independent standalone question.
- Expand common medical abbreviations where unambiguous (IUFD -> Intrauterine fetal death). If unknown, leave unchanged.
- Only correct spelling/abbreviations when it changes meaning. Report such correction in Correction.
- ALso append Abbreviations in Correction
- If input is chitchat or profanity, append "_chitchat" to Rewritten.
- Keep rewritten concise and medically precise.

Few tricky examples:

-chat_history: ""
classify_label: "FOLLOW_UP"
question: "hello how are you?"
-> Rewritten: "hello how are you?_chitchat", Correction: {{}}

-chat_history: ""
classify_label: "CHITCHAT"
question: "hi"
-> Rewritten: "hi_chitchat", Correction: {{}}

- chat_history: "Assistant: <answer> Would you like to know about malaria prevention?"
  classify_label: "FOLLOW_UP"
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

-chat_history: "<answer> would you like to know about newborn jaundice?"
question: "yes"
-> Rewritten: "Provide details on newborn jaundice", Correction: {{}}

- chat_history: "<answer> would you like to know about newborn jaundice?"
question: "sure but also provide cure"
-> Rewritten: "Provide details on newborn jaundice and its cure", Correction: {{}}

classify :
{classify_label}
chat_history:
{chat_history}
user_message:
{user_message}

NOW REWRITE or return as it is in case of chitchat :
"""
    logging.info("[reformulate] sending reformulation prompt")
    try:
        # Attempt LLM invocation
        try:
            resp = reformulate_llm.invoke([HumanMessage(content=prompt)]).content
            # PRINT: raw reformulation response
            print("[REFORM_RAW]:", resp)
            timer.mark("llm.invoke")
            logging.info(f"[reformulate] Raw LLM response: {resp}")
        except Exception as llm_e:
            logging.error(f"[reformulate] Error during LLM invoke in reformulate_query: {llm_e}")
            logging.exception("[reformulate] Full traceback for LLM invoke error:")
            # If LLM invocation fails, return user_message and empty dict as a fallback
            return user_message, {}

        parsed = safe_json_parse(resp)
        # PRINT: parsed reformulation output
        print("[REFORM_PARSED]:", parsed)
        logging.info(f"[reformulate] Parsed LLM response: {parsed}")
        if parsed:
            # Ensure Correction is always a dictionary
            correction_output = parsed.get("Correction", {})
            if isinstance(correction_output, str):
                correction_output = {}

            rewritten = parsed.get("Rewritten", user_message) or user_message

           
            return rewritten, correction_output
    except Exception as e:
        logging.error(f"[reformulate] General error in reformulate_query: {e}")
        logging.exception("[reformulate] Full traceback for general error:")
    return user_message, {} # Default to empty dict for correction in all failure cases

def handle_chitchat(user_message: str, chat_history: str) -> str:
    prompt = chitchat_prompt.format(conversation=user_message, chat_history=chat_history)
    logging.info("[chitchat] sending to chitchat model")
    try:
        return chitchat_llm.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        logging.error(f"[chitchat] LLM failed: {e}")
        return "Whoa, let’s keep it polite, please! 😊"

# -------------------- MAIN PIPELINE (called by API) --------------------
async def medical_pipeline_api(user_id: str, user_message: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    logging.info(f"[pipeline] Starting medical_pipeline_api for user_id: {user_id}, message: {user_message[:50]}")
    timer = CheckpointTimer("medical_pipeline_api")
    history_pairs, current_summary = await get_history(user_id)
    timer.mark("get_history")
    logging.info(f"[pipeline] Fetched history. Pairs: {len(history_pairs)}, Summary length: {len(current_summary)}")
    
    # We need to reconstruct the chat_history for the LLM prompts
    chat_history_context_for_llm = get_chat_context(history_pairs, current_summary)
    timer.mark("get_chat_context")
    logging.info(f"[pipeline] chat_history (summary+last3) context length: {len(chat_history_context_for_llm)}")

    # 1) classify
    logging.info("[pipeline] Step 1: Classifying message...")
    label, reason = classify_message(chat_history_context_for_llm, user_message)
    logging.info(f"[pipeline] classifier -> {label} ({reason})")

    # If classifier says CHITCHAT, short-circuit: call chitchat LLM and return immediately
    try:
        if isinstance(label, str) and label.strip().upper() == "CHITCHAT":
            logging.info("[pipeline] Classifier detected CHITCHAT — short-circuiting to chitchat LLM.")
            reply = handle_chitchat(user_message, chat_history_context_for_llm)
            # Do not update or save chitchat history per request
            timer.total("request")
            return {"answer": reply, "intent": "chitchat", "follow_up": None}
    except Exception:
        # If something unexpected happens, continue the normal flow
        logging.exception("[pipeline] Error during chitchat short-circuit check; continuing pipeline")

    # 2) reformulate (handles follow-ups, abbrs, corrections)
    logging.info("[pipeline] Step 2: Reformulating query...")
    reformulate_result = reformulate_query(
        chat_history_context_for_llm,
        user_message,
        label # Pass the classified label to reformulation
    )
    timer.mark("reformulate_query")
    logging.info(f"[pipeline] Raw result from reformulate_query: {reformulate_result} (type: {type(reformulate_result)})")
    rewritten, correction = reformulate_result # Unpack here
    logging.info(f"[pipeline] Unpacked rewritten: {rewritten}, correction: {correction}")
    logging.info(f"[pipeline] reformulated: {rewritten}  | correction: {correction}")



    # ---- HYBRID RETRIEVAL ----
    logging.info("[pipeline] Step 3 (Medical): Performing hybrid retrieval...")
    candidates = hybrid_retrieve(rewritten)   # vector + bm25 + rerank
    timer.mark("hybrid_retrieve")
    logging.info(f"[pipeline] retrieved {len(candidates)} re-ranked candidates")

    logging.info("[pipeline] Step 4: Judging sufficiency...")
    judge = judge_sufficiency(rewritten, candidates)
    timer.mark("judge_sufficiency")
    logging.info(
        f"[pipeline] judge selected {len(judge['answer_chunks'])} answer chunks, "
        f"{len(judge['followup_chunks'])} follow-up chunks"
    )

    # ---- JUDGE → SYNTHESIZE ----
    if judge["answer_chunks"]:
        top4 = judge["answer_chunks"]
        followup_candidates = judge["followup_chunks"]

        followup_q = ""
        if followup_candidates: # Add this check
            fc = followup_candidates[0]
            sec = fc["meta"].get("section") if fc.get("meta") else None
            followup_q = sec or (fc["text"])

        logging.info("[pipeline] Step 5: Synthesizing answer...")
        answer = synthesize_answer(rewritten, top4, followup_q,llm)
        timer.mark("synthesize_answer")

        # Apply correction prefix if needed
        if label != "FOLLOW_UP" and correction:
            correction_msg = "I guess you meant " + " and ".join(correction.values())
            answer = correction_msg + "\n" + answer
        # Schedule full update+save in background (do not run update_chat_history in request path)
        logging.info("[pipeline] Scheduling background update+save for answer...")
        background_tasks.add_task(_background_update_and_save, user_id, user_message, answer, "answer", history_pairs, current_summary)
        logging.info("[pipeline] Medical pipeline complete with answer; scheduled background update+save.")
        timer.total("request")
        return {"answer": answer, "intent": "answer", "follow_up": followup_q if followup_q else None}

    else:
        msg = (
            "I apologize, but I do not have sufficient information "
            "in my documents to answer this question accurately."
        )
        # Schedule full update+save in background for no_context
        logging.info("[pipeline] Scheduling background update+save for no_context.")
        background_tasks.add_task(_background_update_and_save, user_id, user_message, msg, "no_context", history_pairs, current_summary)
        logging.info("[pipeline] Medical pipeline complete with no_context; scheduled background update+save.")
        timer.total("request")
        return {"answer": msg, "intent": "no_context", "follow_up": None}


# -------------------- API ENDPOINTS --------------------
@app.on_event("startup")
async def startup_event():
    global embedding_model, reranker, pinecone_index, llm, chitchat_llm, summarizer_llm, reformulate_llm, classifier_llm, EMBED_DIM
    logging.info("Initializing models and Pinecone client...")
    t = CheckpointTimer("startup")
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    t.mark("load_embedding_model")
    EMBED_DIM = embedding_model.get_sentence_embedding_dimension()
    # Use a lightweight ONNX/distilled reranker (FlashRank) for low-latency re-ranking
    try:
        reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
        t.mark("load_reranker")
    except Exception:
        # Fallback to CrossEncoder if flashrank is not available
        reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        t.mark("load_reranker_fallback")

    pc = Pinecone(api_key=PINECONE_KEY)
    t.mark("init_pinecone_client")
    if INDEX_NAME not in pc.list_indexes().names():
        logging.info("Creating Pinecone index (if needed)...")
        pc.create_index(name=INDEX_NAME, dimension=EMBED_DIM, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        t.mark("create_pinecone_index")
    pinecone_index = pc.Index(INDEX_NAME)
    t.mark("pinecone_index_ready")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, api_key=OPENAI_API_KEY)
    chitchat_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=OPENAI_API_KEY)
    summarizer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
    reformulate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
    # Classifier uses a dedicated lightweight model for fast classification
    classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
    t.mark("init_llms")

    await init_db()
    t.mark("init_db")
    logging.info("FastAPI application startup complete.")

@app.get("/chat")
async def chat_endpoint(
    user_id: str = Query(..., description="Unique identifier for the user"),
    message: str = Query(..., description="The user's message or query"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        response = await medical_pipeline_api(user_id, message, background_tasks)
        return response
    except Exception as e:
        logging.error(f"Error in chat endpoint for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

