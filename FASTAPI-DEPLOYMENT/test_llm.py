import os
import json
import time
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
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# -------------------- CONFIG --------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY_NEW")
INDEX_NAME = os.getenv("PINECONE_INDEX", "medical-chatbot-index")
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY missing in environment")

if not OPENAI_API_KEY or not PINECONE_KEY:
    raise RuntimeError("OPENAI_API_KEY or PINECONE_API_KEY_NEW missing in environment")

app = FastAPI()

# Global variables for models and Pinecone index
embedding_model: SentenceTransformer = None
reranker: CrossEncoder = None
pinecone_index: Pinecone.Index = None
llm: ChatOpenAI = None
# chitchat_llm removed - now using Gemini for chitchat
summarizer_llm: ChatOpenAI = None
reformulate_llm: ChatOpenAI = None
classifier_llm: ChatOpenAI = None
gemini_llm: ChatGoogleGenerativeAI = None  # Gemini for ALL LLM tasks
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
        # print timing for this checkpoint
        print(f"[{self.name}] {label} took {elapsed:.3f}s")
        self.last = now
        return elapsed

    def total(self, label: str) -> float:
        """Log total time since timer creation and return it."""
        now = time.perf_counter()
        total_elapsed = now - self.start
        print(f"[{self.name}] TOTAL {label} took {total_elapsed:.3f}s")
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
            print("[DB] Added 'intent' column to chat_history table.")
            
        if 'summary' not in columns:
            await db.execute("ALTER TABLE chat_history ADD COLUMN summary TEXT DEFAULT ''")
            print("[DB] Added 'summary' column to chat_history table.")
            
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
        print("[_background_update_and_save] saved history in background")
    except Exception as e:
        print(f"[_background_update_and_save] failed: {e}")

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

# Combined classification + reformulation prompt
combined_classify_reformulate_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
Return JSON only: {{"classification":"MEDICAL_QUESTION|FOLLOW_UP|CHITCHAT","reason":"short explanation","rewritten_query":"optimized query","corrections":{{"misspelled":"corrected","abbrev":"expanded"}}}}

You are a women's healthcare assistant that performs classification AND query reformulation in one step.

CLASSIFICATION RULES:
- MEDICAL_QUESTION: any standalone question about medical facts, diagnoses, treatments, NEWBORN CARE, or women's health
- FOLLOW_UP: short responses to previous assistant suggestions (yes, sure, prevention please, etc.)
- CHITCHAT: greetings, thanks, smalltalk, profanity, non-medical topics, or explicit "stop" requests OR ANYTHING WHICH IS NON-MEDICAL OR NON_FOLLOWUP

REFORMULATION RULES:
- If CHITCHAT: return original message unchanged (no reformulation needed)
- If MEDICAL_QUESTION: rewrite as "Give information about [topic] and cure if available" format
- If FOLLOW_UP: expand into full standalone question using chat history context
- Correct spelling/abbreviations when meaning changes
- Expand medical abbreviations (IUFD -> Intrauterine fetal death)
- Keep rewritten_query concise and medically precise

EXAMPLES:

chat_history: Assistant: Would you like to know about jaundice?
question: clarify its types
-> {{"classification":"FOLLOW_UP","reason":"asking for clarification about previous answer","rewritten_query":"What are the different types of jaundice?","corrections":{{}}}}

chat_history: Assistant: Would you like to know about jaundice?
question: "clarify the dosoge and types of amilicin"
-> {{"classification":"MEDICAL_QUESTION","reason":"new question not related to previous suggestion","rewritten_query":"Give information about amoxicillin dosage and types","corrections":{{"dosoge":"dosage","amilicin":"amoxicillin"}}}}

chat_history: "Assistant: Would you like to know about jaundice?"
question: "yes"
-> {{"classification":"FOLLOW_UP","reason":"affirmation to assistant suggestion","rewritten_query":"Provide details on jaundice","corrections":{{}}}}

chat_history: "Assistant: Would you like to know about jaundice?"
question: "no"
-> {{"classification":"CHITCHAT","reason":"user declining assistant suggestion","rewritten_query":"no","corrections":{{}}}}

chat_history: ""
question: "how to feed a newborn?"
-> {{"classification":"MEDICAL_QUESTION","reason":"asking about newborn care","rewritten_query":"Give information about newborn feeding and care","corrections":{{}}}}

chat_history: ""
question: "hi, how are you?"
-> {{"classification":"CHITCHAT","reason":"greeting","rewritten_query":"hi, how are you?","corrections":{{}}}}

chat_history: ""
question: "when to bath my new born?"
-> {{"classification":"MEDICAL_QUESTION","reason":"asking about newborn care","rewritten_query":"Give information about newborn bathing and care","corrections":{{"bath":"bathe"}}}}

chat_history: ""
question: "what causes jaundice?"
-> {{"classification":"MEDICAL_QUESTION","reason":"standalone medical question","rewritten_query":"Give information about jaundice causes and treatment","corrections":{{}}}}

chat_history: ""
question: "baby is not sucking mother's milk"
-> {{"classification":"MEDICAL_QUESTION","reason":"asking about newborn feeding issues","rewritten_query":"Give information about newborn feeding problems and solutions","corrections":{{}}}}

chat_history: ""
question: "show me bitcoin price"
-> {{"classification":"CHITCHAT","reason":"non-medical topic","rewritten_query":"show me bitcoin price","corrections":{{}}}}

chat_history: ""
question: "denger sign for johndice"
-> {{"classification":"MEDICAL_QUESTION","reason":"medical question with misspellings","rewritten_query":"What are the danger signs for jaundice?","corrections":{{"denger":"danger","johndice":"jaundice"}}}}

chat_history: ""
question: "depression"
-> {{"classification":"MEDICAL_QUESTION","reason":"single medical term","rewritten_query":"Give information about depression and cure if available","corrections":{{}}}}

chat_history: "Assistant: Would you like to know about postpartum hemorrhage?"
question: "any treatment?"
-> {{"classification":"FOLLOW_UP","reason":"follow-up about previous topic","rewritten_query":"What are treatments for postpartum hemorrhage?","corrections":{{}}}}

chat_history: "<answer> would you like to know about newborn jaundice?"
question: "yes"
-> {{"classification":"FOLLOW_UP","reason":"affirmation to suggestion","rewritten_query":"Provide details on newborn jaundice","corrections":{{}}}}

chat_history: "<answer> would you like to know about newborn jaundice?"
question: "sure but also provide cure"
-> {{"classification":"FOLLOW_UP","reason":"affirmation with additional request","rewritten_query":"Provide details on newborn jaundice and its cure","corrections":{{}}}}

Now analyze and respond:
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
Instructions:





Respond to the user’s input ({conversation}) with a friendly, professional, and empathetic tone, avoiding technical expertise.



Maintain a conversational style, steering users toward women’s health-related medical questions without explicitly listing capabilities or limitations (e.g., avoid saying "I am not a doctor").



Unless distress is detected (e.g., suicidal intent), use a light, engaging tone to encourage medical queries.



For distress signals, provide generic, politically correct advice and gently redirect to medical questions.



Avoid self-referential comments about tone (e.g., "I am chirpy") and focus on the user’s input.



If medical advice is requested, politely decline and redirect to medical questions without technical input.

  Examples: 





User: "How are you doing?"
Bot: "I’m doing wonderfully, thank you! How about you? I’d be delighted to assist with any women’s health questions you might have."



User: "Who is your favorite cricketer?"
Bot: "I appreciate the interest, but I’m here to support you with women’s health topics. Feel free to ask me anything related!"



User: "Hey bitch, how are you?"
Bot: "Please let’s keep this respectful. I’m here to help with women’s health-related queries—how can I assist you today?"



User: "I want to suicide."
Bot: "I’m truly sorry to hear you’re feeling this way. Please seek professional medical help. I’m available to support you with women’s health questions—let me know how I can assist."



User: "What is the temperature today in Pune?"
Bot: "I can’t help with that, but I’d love to assist with women’s health topics. Do you have any questions in that area?"



User: "How do I treat a cold?"
Bot: "I’m not able to provide medical advice, but I’d be happy to help with women’s health-related questions. What else can I assist you with?"



User: "Tell me about pregnancy."
Bot: "I can’t offer medical advice, but I’m here to guide you toward women’s health topics. Would you like information on related subjects?"

Edge Cases:





User: "Goodbye."
Bot: "Take care! I’m here whenever you’d like to discuss women’s health—feel free to return!"



User: "You’re useless!"
Bot: "I’m sorry you feel that way. I’m designed to assist with women’s health questions—perhaps there’s something I can help with there?"



User: "I feel so alone."
Bot: "I’m sorry you’re feeling that way. Please consider reaching out for support. I can assist with women’s health topics—how else may I help?"

Response Format:

Provide a single, concise reply based on the above guidelines.

Conversation: {conversation} Reply:"""
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
    - Be Very PRECISE AND TO THE POINT ABOUT WHAT IS ASKED IN THE QUERY AND ANSWER SHOULD BE STRICTLY IN THE CONTEXT PROVIDED AND NOT ANYTHING ELSE.
    - ANSWER STRICTLY IN 150-200 WORDS (USING BULLET AND SUB-BULLET POINTS [MAX 5-6 POINTS]) which encapusaltes the key points and information from the context to answer the query in an APT FLOW
    - Always follow below mentioned rules at all times : 
            • Begin each answer with: **"According to <source>"** (extract filename from metadata if available **DONT MENTION EXTENSIONS** eg, According to abc✅(correct), According to xyz.pdf❌(incorrect)). 
            • Answer concisely in bullet and sub-bullet points. 
            • Add 4-5 bullet points if sufficient information is available in context to answer the query, otherwise minimum of 3 should be used
            • Summarize meaningfully and in a perfect flow
            • Each bullet must be factual and context-bound. 
    - Answer the best answer that can be framed from the context to the query and dont mentions referance to what's not-there in the context (for eg . the document doesn't have much info about <topic> ❌ [these kind of sentence refering about the doc other than the source is not needed])
    - Respect chat history for coherence. 
    - Always include a follow up question 
         - if <context_followup> is non-empty in the format(without bullet points) "Would you like to know about <a follow up question STRICTLY from context_followup not overlapping with the answer generated>?"
         - The genrated follow up should be about a medical topic and not any general topic(eg.Clinical Reference Manual for Advanced Neonatal Care in Ethiopia?" ❌) STRICTLY from context_followup
    - THE FRAMING OF ANSWER SHOULD BE AT PROFESSIONAL EXPERT OF ENGLISH AND THE PERFECT FLOW OF ALL INFORMATION SHOULD MAKE SENSE AND NOT BE RANDOM SPITTING OF INFORMATION.
    - STRICTLY ADHERE TO THE WORD LIMIT AND BULLET POINT RULES and not fetching any information from the web or general knowledge or prior knowledge or any other sources.

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
    print("="*60)
    print("HYBRID RETRIEVAL BLOCK BREAKDOWN")
    print("="*60)
    print(f"[HYBRID] Query: {query[:50]}...")
    print(f"[HYBRID] top_k_vec={top_k_vec}, u_cap={u_cap}")
    
    timer = CheckpointTimer("hybrid_retrieve")
    
    # 1) Vector search - EMBEDDING ENCODING
    print("[HYBRID] Step 1: Encoding query with embedding model...")
    q_emb = embedding_model.encode(query, convert_to_numpy=True).tolist()
    timer.mark("embedding.encode")
    print(f"[HYBRID] embedding.encode took {timer.last - timer.start:.3f}s")
    
    # 2) PINECONE VECTOR SEARCH
    print("[HYBRID] Step 2: Querying Pinecone vector database...")
    vec_resp = pinecone_index.query(
        vector=q_emb,
        top_k=top_k_vec,
        include_metadata=True,
        include_values=False
    )
    timer.mark("pinecone.query")
    print(f"[HYBRID] pinecone.query took {timer.last - timer.start:.3f}s")
    print(f"[HYBRID] Retrieved {len(vec_resp.matches)} matches from Pinecone")

    # 3) PROCESS MATCHES
    print("[HYBRID] Step 3: Processing matches and deduplicating...")
    vec_matches = [{"id": m.id, "text": m.metadata.get("text_full", m.metadata.get("text_snippet", "")), "meta": m.metadata} for m in vec_resp.matches]
    timer.mark("process_matches")
    print(f"[HYBRID] process_matches took {timer.last - timer.start:.3f}s")

    # 4) COMBINE & CAP
    candidates = {m["id"]: m for m in (vec_matches)}  # deduplicate
    candidates = list(candidates.values())[:u_cap]
    timer.mark("deduplicate_cap")
    print(f"[HYBRID] deduplicate_cap took {timer.last - timer.start:.3f}s")
    print(f"[HYBRID] Final candidates: {len(candidates)}")

    # 5) RE-RANKING
    print("[HYBRID] Step 4: Re-ranking candidates...")
    if candidates:
        # If FlashRank Ranker is available, it exposes a `rerank` method
        if hasattr(reranker, "rerank"):
            print("[HYBRID] Using FlashRank reranker...")
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
                print(f"[hybrid_retrieve] FlashRank rerank failed: {e}")
                # fallback: set neutral scores so later steps can handle
                for c in candidates:
                    c["scores"] = {"cross": 0.0}
            timer.mark("flashrank.rerank")
            print(f"[HYBRID] flashrank.rerank took {timer.last - timer.start:.3f}s")
        else:
            # Fallback: existing CrossEncoder.predict path
            print("[HYBRID] Using CrossEncoder reranker...")
            pairs = [(query, c["text"]) for c in candidates]
            try:
                scores = reranker.predict(pairs)
                for i, c in enumerate(candidates):
                    c["scores"] = {"cross": float(scores[i])}
            except Exception as e:
                print(f"[hybrid_retrieve] CrossEncoder predict failed: {e}")
                for c in candidates:
                    c["scores"] = {"cross": 0.0}
            timer.mark("cross-encoder.predict")
            print(f"[HYBRID] cross-encoder.predict took {timer.last - timer.start:.3f}s")
    else:
        # no candidates
        print("[HYBRID] No candidates found")
        timer.mark("no_candidates")

    # 6) SORT CANDIDATES
    print("[HYBRID] Step 5: Sorting candidates by relevance score...")
    candidates = sorted(candidates, key=lambda x: x.get("scores", {}).get("cross", 0.0), reverse=True)
    timer.mark("sort_candidates")
    print(f"[HYBRID] sort_candidates took {timer.last - timer.start:.3f}s")
    
    # FINAL SUMMARY
    total_time = timer.total("hybrid_retrieve")
    print(f"[HYBRID] TOTAL HYBRID RETRIEVAL TIME: {total_time:.3f}s")
    print(f"[HYBRID] Returning {len(candidates)} ranked candidates")
    print("="*60)
    
    return candidates

# -------------------- JUDGE + ANSWER --------------------
def judge_sufficiency(query: str, candidates: List[Dict[str, Any]], judge_llm: ChatOpenAI | None = None, threshold_weak: float = 0.25) -> Dict[str, List[Dict[str, Any]]]:
    """
    Judge each candidate chunk individually for sufficiency.
    Return the top 4 qualified chunks for answering,
    and next 2 for follow-up suggestion.
    """
    global gemini_llm # Use Gemini for judging instead of OpenAI

    if judge_llm is None: # Use Gemini LLM for judging
        judge_llm = gemini_llm

    qualified_with_scores = [] # Store qualified chunks along with their cross-encoder scores and topic_match
    followup_chunks_raw = [] # Store non-qualified chunks for potential follow-up
    topic_match_order = {"strong": 3, "medium": 2, "absolutely_not_possible": 1}

    print(f"[judge_sufficiency] len of candidates {len(candidates)}")
    print("[judge_sufficiency] Using GEMINI LLM for judging")
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
        #print("[JUDGE_RAW_RESP]:", resp)
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
                    print(f"[judge_sufficiency] Invalid index in LLM judgment: {judgment}")
                    # Fallback for invalid index
                    if c["scores"]["cross"] > threshold_weak:
                        qualified_with_scores.append(c)
                    else:
                        followup_chunks_raw.append(c)
        else:
            print("[judge_sufficiency] LLM did not return valid batched judgments. Falling back to cross-encoder scores.")
            print("[JUDGE_PARSED]: None or invalid format; falling back to cross-encoder scores")
            # Fallback based on cross-encoder score if LLM fails to parse or returns no judgments
            for c in candidates:
                c['meta']['topic_match_score'] = topic_match_order.get("medium", 0) # Assign a default topic_match_score
                if c["scores"]["cross"] > threshold_weak:
                    qualified_with_scores.append(c)
                else:
                    followup_chunks_raw.append(c)
    except Exception as e:
        print(f"[judge_sufficiency] Error during batched LLM judging: {e}")
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
    
    print(f"[judge_sufficiency] BEFORE answer_chunks={len(qualified)} BEFORE followup_chunks={len(followup_chunks_raw)}")
    
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

    print(f"[judge_sufficiency] AFTER answer_chunks={len(answer_chunks)} AFTER followup_chunks={len(followup_chunks)}")
    timer.mark("done")
    return {"answer_chunks": answer_chunks, "followup_chunks": followup_chunks}

def synthesize_answer(query: str, top_candidates: List[Dict[str, Any]], context_followup: str, main_llm: ChatOpenAI | None = None) -> str:
    global gemini_llm # Use Gemini for synthesis instead of OpenAI

    if main_llm is None: # Use Gemini LLM for synthesis
        main_llm = gemini_llm

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
    print("[answer] sending synthesis prompt to GEMINI LLM")
    timer = CheckpointTimer("synthesize_answer")
    resp = main_llm.invoke([HumanMessage(content=prompt)]).content
    timer.mark("synthesize_llm.invoke")

    return resp

# -------------------- CLASSIFY / REFORMULATE / CHITCHAT --------------------
def classify_message(chat_history: str, user_message: str) -> Tuple[str, str]:
    prompt = classifier_prompt.format(chat_history=chat_history, question=user_message)
    print("[classify] sending classification prompt")
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
        print(f"[classify] LLM classify failed: {e}")

    # # fallback heuristics
    # low = user_message.lower().strip()
    # if low in ("yes", "sure", "ok", "yep") and "Would you like to know about" in chat_history:
    #     return "FOLLOW_UP", "affirmation heuristic"
    # if any(w in low for w in ("hi", "hello", "thanks", "thank you", "bye")):
    #     return "CHITCHAT", "greeting heuristic"

def reformulate_query(chat_history: str, user_message: str, classify_label: str) -> Tuple[str, Dict[str, str]]:
    global llm # Access global llm
    print("[reformulate] start")
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
    print("[reformulate] sending reformulation prompt")
    try:
        # Attempt LLM invocation
        try:
            resp = reformulate_llm.invoke([HumanMessage(content=prompt)]).content
            # PRINT: raw reformulation response
            print("[REFORM_RAW]:", resp)
            timer.mark("llm.invoke")
            print(f"[reformulate] Raw LLM response: {resp}")
        except Exception as llm_e:
            print(f"[reformulate] Error during LLM invoke in reformulate_query: {llm_e}")
            # If LLM invocation fails, return user_message and empty dict as a fallback
            return user_message, {}

        parsed = safe_json_parse(resp)
        # PRINT: parsed reformulation output
        print("[REFORM_PARSED]:", parsed)
        print(f"[reformulate] Parsed LLM response: {parsed}")
        if parsed:
            # Ensure Correction is always a dictionary
            correction_output = parsed.get("Correction", {})
            if isinstance(correction_output, str):
                correction_output = {}

            rewritten = parsed.get("Rewritten", user_message) or user_message

           
            return rewritten, correction_output
    except Exception as e:
        print(f"[reformulate] General error in reformulate_query: {e}")
    return user_message, {} # Default to empty dict for correction in all failure cases

def classify_and_reformulate(chat_history: str, user_message: str) -> Tuple[str, str, str, Dict[str, str]]:
    """
    Combined classification and reformulation using Gemini in a single LLM call.
    Returns: (classification, reason, rewritten_query, corrections)
    """
    global gemini_llm
    
    print("="*60)
    print("CLASSIFICATION + REFORMULATION BLOCK")
    print("="*60)
    
    # Start timing for this block
    block_start = time.perf_counter()
    
    prompt = combined_classify_reformulate_prompt.format(chat_history=chat_history, question=user_message)
    print(f"[CLASSIFY_REFORM] Input: {user_message[:100]}...")
    print(f"[CLASSIFY_REFORM] Chat history length: {len(chat_history)}")
    print("[CLASSIFY_REFORM] Sending prompt to Gemini...")
    
    try:
        # Time the LLM call specifically
        llm_start = time.perf_counter()
        resp = gemini_llm.invoke([HumanMessage(content=prompt)]).content
        llm_end = time.perf_counter()
        
        print(f"[CLASSIFY_REFORM] Gemini LLM call took {llm_end - llm_start:.3f} seconds")
        
        # Print raw response for debugging
        print("[CLASSIFY_REFORM] Raw Gemini response:")
        print("-" * 40)
        print(resp)
        print("-" * 40)
        
        # Parse JSON response
        parse_start = time.perf_counter()
        parsed = safe_json_parse(resp)
        parse_end = time.perf_counter()
        
        print(f"[CLASSIFY_REFORM] JSON parsing took {parse_end - parse_start:.3f} seconds")
        print("[CLASSIFY_REFORM] Parsed JSON:")
        print("-" * 40)
        print(parsed)
        print("-" * 40)
        
        if parsed:
            classification = parsed.get("classification", "MEDICAL_QUESTION")
            reason = parsed.get("reason", "")
            rewritten_query = parsed.get("rewritten_query", user_message)
            corrections = parsed.get("corrections", {})
            
            # Ensure corrections is always a dict
            if not isinstance(corrections, dict):
                corrections = {}
            
            # For CHITCHAT, use original message unchanged
            if classification.upper() == "CHITCHAT":
                print("[CLASSIFY_REFORM] CHITCHAT detected - using original message unchanged")
                rewritten_query = user_message
                corrections = {}
            
            # Print detailed outputs
            print("[CLASSIFY_REFORM] EXTRACTED RESULTS:")
            print(f"  Classification: {classification}")
            print(f"  Reason: {reason}")
            print(f"  Rewritten Query: {rewritten_query}")
            print(f"  Corrections: {corrections}")
            
            block_end = time.perf_counter()
            print(f"[CLASSIFY_REFORM] TOTAL BLOCK TIME: {block_end - block_start:.3f} seconds")
            print("="*60)
            
            return classification, reason, rewritten_query, corrections
        else:
            print("[CLASSIFY_REFORM] Failed to parse JSON response")
            
    except Exception as e:
        print(f"[CLASSIFY_REFORM] Gemini LLM call failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback: return original message with default classification
    block_end = time.perf_counter()
    print(f"[CLASSIFY_REFORM] FALLBACK - TOTAL BLOCK TIME: {block_end - block_start:.3f} seconds")
    print("="*60)
    return "MEDICAL_QUESTION", "fallback", user_message, {}

def handle_chitchat(user_message: str, chat_history: str) -> str:
    global gemini_llm
    prompt = chitchat_prompt.format(conversation=user_message, chat_history=chat_history)
    print("[chitchat] sending to GEMINI chitchat model")
    try:
        return gemini_llm.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        print(f"[chitchat] Gemini LLM failed: {e}")
        return "Whoa, let's keep it polite, please! 😊"

# -------------------- VIDEO MATCHING SYSTEM (SIMPLIFIED BERT APPROACH) --------------------
class VideoMatchingSystem:
    def __init__(self, video_file_path: str = "./FILES/video_link_topic.xlsx"):
        """Initialize the simplified video matching system using BERT similarity"""
        self.video_file_path = video_file_path
        self.topic_list = []  # List of topic strings
        self.url_list = []    # List of corresponding URLs
        
        # Load video data
        self._load_video_data()
        
        # Initialize BERT model for similarity
        print("[VIDEO_SYSTEM] Loading BERT model for semantic similarity...")
        self.similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("[VIDEO_SYSTEM] BERT model loaded successfully")
    
    def _load_video_data(self):
        """Load and preprocess video data into simple lists using Description column"""
        try:
            df = pd.read_excel(self.video_file_path)
            print(f"[VIDEO_SYSTEM] Loaded {len(df)} videos from {self.video_file_path}")
            
            # Create simple lists using Description column instead of video_topic
            for idx, row in df.iterrows():
                description = row['Description'].strip()
                url = row['URL'].strip()
                
                if description and url:
                    self.topic_list.append(description)
                    self.url_list.append(url)
            
            print(f"[VIDEO_SYSTEM] Created description_list with {len(self.topic_list)} descriptions")
            print(f"[VIDEO_SYSTEM] Sample descriptions:")
            for i, desc in enumerate(self.topic_list[:3]):
                print(f"  {i}: {desc[:100]}...")
            
        except Exception as e:
            print(f"[VIDEO_SYSTEM] Error loading video data: {e}")
            self.topic_list = []
            self.url_list = []
    
    def find_relevant_video(self, answer: str) -> Optional[str]:
        """Find relevant video using BERT similarity + LLM verification"""
        if not self.topic_list:
            return None
        
        print(f"[VIDEO_SYSTEM] Searching for video for answer: {answer[:100]}...")
        
        # Step 1: BERT Semantic Similarity
        print("[VIDEO_SYSTEM] Step 1: Computing BERT semantic similarities...")
        bert_start = time.perf_counter()
        
        # Encode answer and all topics
        answer_embedding = self.similarity_model.encode([answer])
        topic_embeddings = self.similarity_model.encode(self.topic_list)
        
        # Compute cosine similarities
        similarities = cosine_similarity(answer_embedding, topic_embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        bert_end = time.perf_counter()
        print(f"[VIDEO_SYSTEM] BERT similarity computation took {bert_end - bert_start:.3f} seconds")
        print(f"[VIDEO_SYSTEM] Best similarity score: {best_similarity:.3f}")
        print(f"[VIDEO_SYSTEM] Best description: {self.topic_list[best_idx][:100]}...")
        
        # Step 2: LLM Verification (only for top match)
        if best_similarity >= 0.50:  # Threshold for semantic similarity
            print("[VIDEO_SYSTEM] Step 2: LLM verification of top match...")
            llm_start = time.perf_counter()
            
            verification_result = self._verify_with_llm(answer, self.topic_list[best_idx])
            
            llm_end = time.perf_counter()
            print(f"[VIDEO_SYSTEM] LLM verification took {llm_end - llm_start:.3f} seconds")
            print(f"[VIDEO_SYSTEM] LLM verification result: {verification_result}")
            
            if verification_result:
                video_url = self.url_list[best_idx]
                print(f"[VIDEO_SYSTEM] Found relevant video: {video_url}")
                return video_url
            else:
                print("[VIDEO_SYSTEM] LLM verification failed - no video")
                return None
        else:
            print(f"[VIDEO_SYSTEM] Similarity score {best_similarity:.3f} below threshold 0.50 - no video")
            return None
    
    def _verify_with_llm(self, answer: str, description: str) -> bool:
        """Use Gemini to verify if the video description is contextually relevant to the answer"""
        prompt = f"""Analyze if the video description majorly aligns with the medical answer with referance to the below rules



Question: Is this video description DIRECTLY and STRONGLY related to the medical answer?

Rules:
- Return "YES" only if the video description directly addresses the same medical condition, procedure, or treatment mentioned in the answer
- Return "NO" if the description is related but not directly relevant (e.g., general care vs specific procedure)
- Return "NO" if the description is about a different medical condition entirely

Examples:
- Answer about "eye care for newborns" + Description "video about applying eye medication to prevent infections" → YES
- Answer about "eye care for newborns" + Description "video about umbilical cord care procedures" → NO
- Answer about "temperature measurement" + Description "video about using thermometer to check baby temperature" → YES
Medical Answer: ```{answer}```

Video Description: ```{description}```
Response (YES/NO only):"""

        try:
            response = gemini_llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
            print(f"[VIDEO_SYSTEM] LLM response: {response}")
            return response == "YES"
        except Exception as e:
            print(f"[VIDEO_SYSTEM] LLM verification failed: {e}")
            return False

# -------------------- CACHE SYSTEM (BERT + LLM APPROACH) --------------------
class CacheSystem:
    def __init__(self, cache_file_path: str = "./FILES/cache_questions.xlsx"):
        """Initialize the cache system using BERT similarity + LLM verification"""
        self.cache_file_path = cache_file_path
        self.question_list = []  # List of cached questions
        self.answer_list = []    # List of corresponding answers
        
        # Load cache data
        self._load_cache_data()
        
        # Initialize BERT model for similarity (reuse video system's model)
        if self.question_list:
            print("[CACHE_SYSTEM] Loading BERT model for cache similarity...")
            self.similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("[CACHE_SYSTEM] BERT model loaded successfully")
        else:
            self.similarity_model = None
    
    def _load_cache_data(self):
        """Load cache data from Excel file"""
        try:
            df = pd.read_excel(self.cache_file_path)
            print(f"[CACHE_SYSTEM] Loaded {len(df)} cached Q&A pairs from {self.cache_file_path}")
            
            # Create simple lists
            for idx, row in df.iterrows():
                question = row['question'].strip()
                answer = row['answer'].strip()
                
                if question and answer:
                    self.question_list.append(question)
                    self.answer_list.append(answer)
            
            print(f"[CACHE_SYSTEM] Created cache with {len(self.question_list)} questions")
            print(f"[CACHE_SYSTEM] Sample cached questions:")
            for i, q in enumerate(self.question_list[:3]):
                print(f"  {i}: {q[:80]}...")
            
        except Exception as e:
            print(f"[CACHE_SYSTEM] Error loading cache data: {e}")
            self.question_list = []
            self.answer_list = []
    
    def check_cache(self, reformulated_query: str) -> Optional[str]:
        """Check if reformulated query matches any cached question using BERT + LLM verification"""
        if not self.question_list or not self.similarity_model:
            print("[CACHE_SYSTEM] No cache data or model available")
            return None
        
        print("="*60)
        print("CACHE SYSTEM BLOCK")
        print("="*60)
        print(f"[CACHE_SYSTEM] Checking cache for: {reformulated_query[:100]}...")
        
        # Step 1: BERT Semantic Similarity
        print("[CACHE_SYSTEM] Step 1: Computing BERT semantic similarities...")
        cache_start = time.perf_counter()
        
        # Encode reformulated query and all cached questions
        query_embedding = self.similarity_model.encode([reformulated_query])
        question_embeddings = self.similarity_model.encode(self.question_list)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, question_embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        cache_end = time.perf_counter()
        print(f"[CACHE_SYSTEM] BERT similarity computation took {cache_end - cache_start:.3f} seconds")
        print(f"[CACHE_SYSTEM] Best similarity score: {best_similarity:.3f}")
        print(f"[CACHE_SYSTEM] Best cached question: {self.question_list[best_idx][:100]}...")
        
        # Step 2: Combined LLM Verification + Reframing (only for top match)
        if best_similarity >= 0.60:  # Higher threshold for cache (more strict)
            print("[CACHE_SYSTEM] Step 2: Combined LLM verification and reframing...")
            llm_start = time.perf_counter()
            
            result = self._verify_and_reframe_cache(reformulated_query, self.question_list[best_idx], self.answer_list[best_idx])
            
            llm_end = time.perf_counter()
            print(f"[CACHE_SYSTEM] Combined LLM verification and reframing took {llm_end - llm_start:.3f} seconds")
            
            if result:
                print(f"[CACHE_SYSTEM] Cache HIT! Returning reframed answer")
                print("="*60)
                return result
            else:
                print("[CACHE_SYSTEM] LLM verification failed - cache miss")
                print("="*60)
                return None
        else:
            print(f"[CACHE_SYSTEM] Similarity score {best_similarity:.3f} below threshold 0.60 - cache miss")
            print("="*60)
            return None
    
    def _verify_and_reframe_cache(self, reformulated_query: str, cached_question: str, cached_answer: str) -> Optional[str]:
        """Combined verification and reframing: Check if cached answer can answer the query, and reframe if yes"""
        prompt = f"""You are a medical assistant. Your task is to determine if the provided answer can address the user's query, and if yes, reframe it to directly answer the query.

User Query: {reformulated_query}

Original Question (for reference): {cached_question}

Provided Answer: {cached_answer}

CRITICAL INSTRUCTIONS:

• VERIFICATION STEP:
  - Determine if the provided answer contains information that can answer the user query
  - If YES: Proceed to reframing
  - If NO: Return exactly "NULL" (nothing else)

• REFRAMING RULES (if applicable):
  - Use ONLY information from the provided answer - NO external knowledge
  - Maintain ALL medical facts exactly as they appear in the provided answer
  - Adjust tense, flow, and structure to directly match the user query
  - Do NOT add any external information not present in the provided answer
  - Do NOT remove any medical information from the provided answer
  - Keep the answer concise and directly relevant to the query

• RESPONSE FORMAT REQUIREMENTS (CRITICAL):
  - Provide ONLY the reframed answer - nothing else
  - Do NOT mention "cache", "cached response", "cached answer", or any similar terms
  - Do NOT add prefixes like "The following question can be answered from...", "Based on cached response...", etc.
  - Do NOT add any metadata, explanations, or disclaimers about the source
  - Answer as if you are directly responding to the user's query naturally
  - Start directly with the answer content

EXAMPLES OF DO'S AND DON'TS:

DON'T (WRONG):
❌ "The following question can be answered from cached response: Fever is caused by..."
❌ "Based on the cached answer, fever is caused by..."
❌ "This can be answered from cache: Fever is caused by..."
❌ "Cached response: Fever is caused by..."

DO (CORRECT):
✅ "Fever is caused by infections, inflammatory conditions, and certain medications..."
✅ "Jaundice treatment includes addressing the underlying cause, which may involve..."
✅ "Signs of dehydration include dry mouth, decreased urine output, fatigue..."

VERIFICATION EXAMPLES:
- Query: "What causes fever?" + Answer: "Fever symptoms include..." → NULL (no cause info)
- Query: "What causes fever?" + Answer: "Fever is caused by infections..." → Reframe to focus on causes
- Query: "How to treat jaundice?" + Answer: "Jaundice symptoms are..." → NULL (no treatment info)
- Query: "What are signs of dehydration?" + Answer: "Dehydration signs include..." → Reframe to focus on signs

Response: Return ONLY the reframed answer (with no cache mentions) if applicable, or exactly "NULL" if not applicable."""

        try:
            response = gemini_llm.invoke([HumanMessage(content=prompt)]).content.strip()
            print(f"[CACHE_SYSTEM] LLM response preview: {response[:200]}...")
            
            # Check if LLM returned NULL
            if response.upper().strip() == "NULL":
                print(f"[CACHE_SYSTEM] LLM determined cached answer cannot answer the query")
                return None
            
            # Return the reframed answer
            print(f"[CACHE_SYSTEM] LLM provided reframed answer")
            return response
            
        except Exception as e:
            print(f"[CACHE_SYSTEM] Combined verification and reframing failed: {e}")
            # Fallback: return None to trigger RAG pipeline
            print(f"[CACHE_SYSTEM] Falling back to RAG pipeline")
            return None

# Global cache system
cache_system: CacheSystem = None

# -------------------- MAIN PIPELINE (called by API) --------------------
async def medical_pipeline_api(user_id: str, user_message: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    print(f"[pipeline] Start user_id={user_id}, message={user_message[:50]}")
    timer = CheckpointTimer("medical_pipeline_api")
    start_time = time.perf_counter()

    history_pairs, current_summary = await get_history(user_id)
    t1 = time.perf_counter()
    timer.mark("get_history")
    print(f"history fetch took {t1 - start_time:.2f} secs")
    print(f"[pipeline] History pairs={len(history_pairs)} summary_len={len(current_summary)}")
    
    # We need to reconstruct the chat_history for the LLM prompts
    chat_history_context_for_llm = get_chat_context(history_pairs, current_summary)
    t2 = time.perf_counter()
    timer.mark("get_chat_context")
    print(f"chat context build took {t2 - t1:.2f} secs")
    print(f"[pipeline] chat_context_len={len(chat_history_context_for_llm)}")

    # Combined classification + reformulation using Gemini (OPTIMIZED: Single LLM call instead of two)
    print("\n" + "="*80)
    print("STEP 1+2: CLASSIFICATION + REFORMULATION (GEMINI OPTIMIZED)")
    print("="*80)
    print(f"[PIPELINE] Starting classify_and_reformulate for: {user_message[:50]}...")
    
    label, reason, rewritten, correction = classify_and_reformulate(chat_history_context_for_llm, user_message)
    t3 = time.perf_counter()
    
    print(f"[PIPELINE] classify_and_reformulate took {t3 - t2:.2f} secs")
    print(f"[PIPELINE] FINAL RESULTS:")
    print(f"  Classification: {label}")
    print(f"  Reason: {reason}")
    print(f"  Rewritten Query: {rewritten}")
    print(f"  Corrections: {correction}")
    print("="*80)

    # If classifier says CHITCHAT, short-circuit: call chitchat LLM and return immediately
    # Use ORIGINAL user_message, NOT the rewritten version
    if isinstance(label, str) and label.strip().upper() == "CHITCHAT":
        print("[pipeline] CHITCHAT detected -> chitchat handler")
        print(f"[pipeline] Using ORIGINAL message for chitchat: {user_message}")
        reply = handle_chitchat(user_message, chat_history_context_for_llm)
        # Do not update or save chitchat history per request
        t_end = time.perf_counter()
        timer.total("request")
        print(f"total took {t_end - start_time:.2f} secs")
        return {"answer": reply, "intent": "chitchat", "follow_up": None}

    # Skip the old reformulation step since it's now combined above

    # ---- CACHE CHECK (NEW: BERT + LLM APPROACH) ----
    print("[pipeline] Step 3: Cache check using BERT + LLM verification...")
    cached_answer = None
    if cache_system:
        cache_start = time.perf_counter()
        cached_answer = cache_system.check_cache(rewritten)
        cache_end = time.perf_counter()
        print(f"[pipeline] Cache check took {cache_end - cache_start:.3f} seconds")
    
    if cached_answer:
        print("[pipeline] CACHE HIT! Skipping RAG pipeline...")
        
        # Apply correction prefix if needed
        if label != "FOLLOW_UP" and correction:
            correction_msg = "I guess you meant " + " and ".join(correction.values())
            cached_answer = correction_msg + "\n" + cached_answer
        
        # Find relevant video URL for cached answer
        print("[pipeline] Step 4: Finding relevant video for cached answer...")
        video_url = None
        if video_system:
            video_start = time.perf_counter()
            video_url = video_system.find_relevant_video(cached_answer)
            video_end = time.perf_counter()
            print(f"[pipeline] Video matching took {video_end - video_start:.3f} secs")
            if video_url:
                print(f"[pipeline] Found relevant video: {video_url}")
            else:
                print("[pipeline] No relevant video found")
        
        # Schedule background save for cached answer
        print("[pipeline] schedule background save: cached_answer")
        background_tasks.add_task(_background_update_and_save, user_id, user_message, cached_answer, "answer", history_pairs, current_summary)
        print("[pipeline] done with cached answer")
        t_end = time.perf_counter()
        timer.total("request")
        print(f"total took {t_end - start_time:.2f} secs")
        
        # Return cached response with video URL
        response = {"answer": cached_answer, "intent": "answer", "follow_up": None}
        if video_url:
            response["video_url"] = video_url
        else:
            response["video_url"] = None
        
        return response
    
    print("[pipeline] CACHE MISS! Proceeding with RAG pipeline...")

    # ---- HYBRID RETRIEVAL ----
    print("[pipeline] Step 4: hybrid_retrieve")
    candidates = hybrid_retrieve(rewritten)   # vector + bm25 + rerank
    t4 = time.perf_counter()
    timer.mark("hybrid_retrieve")
    print(f"retrieval took {t4 - t3:.2f} secs")
    print(f"[pipeline] retrieved {len(candidates)} candidates")

    print("[pipeline] Step 5: judge_sufficiency")
    judge = judge_sufficiency(rewritten, candidates)
    t5 = time.perf_counter()
    timer.mark("judge_sufficiency")
    print(f"judging took {t5 - t4:.2f} secs")
    print(f"[pipeline] judge -> answers={len(judge['answer_chunks'])} followups={len(judge['followup_chunks'])}")

    # ---- JUDGE → SYNTHESIZE ----
    if judge["answer_chunks"]:
        top4 = judge["answer_chunks"]
        followup_candidates = judge["followup_chunks"]

        followup_q = ""
        if followup_candidates: # Add this check
            fc = followup_candidates[0]
            sec = fc["meta"].get("section") if fc.get("meta") else None
            followup_q = sec or (fc["text"])

        print("[pipeline] Step 6: synthesize_answer")
        answer = synthesize_answer(rewritten, top4, followup_q, gemini_llm)
        t6 = time.perf_counter()
        timer.mark("synthesize_answer")
        print(f"synthesis took {t6 - t5:.2f} secs")

        # Apply correction prefix if needed
        if label != "FOLLOW_UP" and correction:
            correction_msg = "I guess you meant " + " and ".join(correction.values())
            answer = correction_msg + "\n" + answer
        
        # Find relevant video URL
        print("[pipeline] Step 7: Finding relevant video...")
        video_url = None
        if video_system:
            video_start = time.perf_counter()
            video_url = video_system.find_relevant_video(answer)
            video_end = time.perf_counter()
            print(f"[pipeline] Video matching took {video_end - video_start:.3f} secs")
            if video_url:
                print(f"[pipeline] Found relevant video: {video_url}")
            else:
                print("[pipeline] No relevant video found")
        
        # Schedule full update+save in background (do not run update_chat_history in request path)
        print("[pipeline] schedule background save: answer")
        background_tasks.add_task(_background_update_and_save, user_id, user_message, answer, "answer", history_pairs, current_summary)
        print("[pipeline] done with answer")
        t_end = time.perf_counter()
        timer.total("request")
        print(f"total took {t_end - start_time:.2f} secs")
        
        # Return response with video URL
        response = {"answer": answer, "intent": "answer", "follow_up": followup_q if followup_q else None}
        if video_url:
            response["video_url"] = video_url
        else:
            response["video_url"] = None
        
        return response

    else:
        msg = (
            "I apologize, but I do not have sufficient information "
            "in my documents to answer this question accurately."
        )
        # Schedule full update+save in background for no_context
        print("[pipeline] schedule background save: no_context")
        background_tasks.add_task(_background_update_and_save, user_id, user_message, msg, "no_context", history_pairs, current_summary)
        print("[pipeline] done with no_context")
        t_end = time.perf_counter()
        timer.total("request")
        print(f"total took {t_end - start_time:.2f} secs")
        
        # Return response with video URL (None for no_context)
        return {"answer": msg, "intent": "no_context", "follow_up": None, "video_url": None}


# -------------------- API ENDPOINTS --------------------
@app.on_event("startup")
async def startup_event():
    global embedding_model, reranker, pinecone_index, llm, summarizer_llm, reformulate_llm, classifier_llm, gemini_llm, EMBED_DIM, video_system, cache_system
    print("[startup] Initializing models and Pinecone client...")
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
        print("[startup] Creating Pinecone index (if needed)...")
        pc.create_index(name=INDEX_NAME, dimension=EMBED_DIM, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        t.mark("create_pinecone_index")
    pinecone_index = pc.Index(INDEX_NAME)
    t.mark("pinecone_index_ready")

    # Initialize OpenAI models
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, api_key=OPENAI_API_KEY)
    # chitchat_llm removed - now using Gemini for chitchat
    summarizer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
    reformulate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
    classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
    
    # Initialize Gemini for ALL LLM tasks (classification, reformulation, judging, synthesis, chitchat)
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=GOOGLE_API_KEY)
    t.mark("init_llms")

    # Initialize cache system
    print("[startup] Initializing cache system...")
    cache_system = CacheSystem()
    t.mark("init_cache_system")

    # Initialize video matching system
    print("[startup] Initializing video matching system...")
    video_system = VideoMatchingSystem()
    t.mark("init_video_system")

    await init_db()
    t.mark("init_db")
    print("[startup] FastAPI application startup complete.")

@app.get("/test-debug")
async def test_debug_endpoint():
    """Test endpoint to verify debug statements are working"""
    print("="*60)
    print("TEST DEBUG ENDPOINT CALLED")
    print("="*60)
    return {"status": "debug_test_working", "message": "Check server console for debug output"}

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
        print(f"[chat_endpoint] Error for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)