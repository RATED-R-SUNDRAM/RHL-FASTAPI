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
# ChromaDB for local vector store
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    Settings = None
    CHROMADB_AVAILABLE = False
    print("WARNING: chromadb not available. Install with: pip install chromadb")
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

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in environment")
# PINECONE_KEY no longer required - using local ChromaDB

app = FastAPI()

# Global variables for models and ChromaDB collection
embedding_model: SentenceTransformer = None
reranker: CrossEncoder = None
chromadb_collection = None  # ChromaDB collection (replaces pinecone_index)
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

def get_chat_context(history_pairs: List[Tuple[str, str, str]], summary: str) -> Tuple[str, str]:
    """
    Returns (full_chat_history, last_3_qa_pairs)
    - full_chat_history: Full formatted history for LLM context
    - last_3_qa_pairs: Last 3 non-chitchat Q&A pairs formatted for FOLLOW_UP reformulation
                      Format: "User: <q1> | Bot: <a1>\nUser: <q2> | Bot: <a2>\nUser: <q3> | Bot: <a3>"
                      This allows LLM to understand conversation context and link follow-ups to the MAIN topic
                      Returns empty string if no non-chitchat pairs exist
    """
    # Filter out chitchat messages from verbatim history when constructing context for LLM
    filtered_verbatim_pairs = [(q, a) for q, a, intent in history_pairs if intent != "chitchat"]
    verbatim = "\n".join([f"User: {q} | Bot: {a}" for q, a in filtered_verbatim_pairs])
    combined = (summary + "\n" + verbatim).strip()
    
    # Extract last 3 non-chitchat Q&A pairs explicitly for FOLLOW_UP reformulation
    # This allows the LLM to understand conversation flow and link follow-ups to the MAIN topic
    last_3_qa_pairs = ""
    if filtered_verbatim_pairs:
        # Take last 3 pairs (or fewer if not available)
        recent_pairs = filtered_verbatim_pairs[-3:]
        
        # Format each pair, extracting follow-up offers from "no_context" responses
        formatted_pairs = []
        for q, a in recent_pairs:
            # Extract follow-up offer from "no_context" responses if present
            if "Would you like to know about" in a or "would you like to know about" in a:
                followup_match = re.search(r'[Ww]ould you like to know about ([^?]+)', a)
                if followup_match:
                    # Format as "Assistant: Would you like to know about X?"
                    formatted_answer = f"Assistant: Would you like to know about {followup_match.group(1)}?"
                else:
                    formatted_answer = f"Assistant: {a}"
            else:
                formatted_answer = f"Assistant: {a}"
            
            formatted_pairs.append(f"User: {q} | {formatted_answer}")
        
        last_3_qa_pairs = "\n".join(formatted_pairs)
    
    return combined, last_3_qa_pairs


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
Return JSON only: {{"classification":"MEDICAL_QUESTION|FOLLOW_UP|CHITCHAT","reason":"short explanation","rewritten_query":"optimized query","corrections":{{"misspelled":"corrected","abbrev":"expanded"}},"sample_answer":"brief sample answer in 100-150 words"}}

You are a women's healthcare assistant that performs classification AND query reformulation in one step.

IMPORTANT: For MEDICAL_QUESTION and FOLLOW_UP (not CHITCHAT), also generate a "sample_answer" field - a brief 350 word sample response that would answer the query and that would be present in medical documents. This should be in answer format (not question format), similar to how medical documents phrase information. Use declarative statements. For CHITCHAT, set sample_answer to empty string "".

CLASSIFICATION RULES:
- MEDICAL_QUESTION: any standalone question about medical facts, diagnoses, treatments, NEWBORN CARE, or women's health
- MEDICAL_QUESTION : Even if user asks multiple times about the same MEDICAL TOPIC, it should be considered as a MEDICAL_QUESTION
- FOLLOW_UP: short responses to previous assistant suggestions (yes, sure, prevention please, etc.)
- CHITCHAT: greetings, thanks, smalltalk, profanity, non-medical topics, or explicit "stop" requests OR ANYTHING WHICH IS NON-MEDICAL OR NON_FOLLOWUP

REFORMULATION RULES:
- If CHITCHAT: return original message unchanged (no reformulation needed)
- If MEDICAL_QUESTION: rewrite as "Give information about [topic] in detail
" format
- If FOLLOW_UP: 
  * The chat_history provided contains the LAST 3 non-chitchat Q&A pairs (format: "User: <q1> | Bot: <a1>\nUser: <q2> | Bot: <a2>\nUser: <q3> | Bot: <a3>")
  * Analyze the conversation flow to identify the MAIN topic being discussed
  * The MAIN topic is typically the FIRST/ORIGINAL topic introduced in the conversation (e.g., "jaundice" in a conversation about jaundice, then cure, then prevention)
  
  * For DIRECT AFFIRMATIONS (yes, sure, okay, etc.):
    - If the most recent Bot response contains "Would you like to know about [topic]?" → reformulate to "Provide details on [topic]"
    - Use the most recent assistant suggestion directly
  
  * For ASPECT-BASED FOLLOW-UPS (prevention, cure, treatment, symptoms, causes, types, etc.):
    - Identify the MAIN topic from the conversation history (the primary medical condition/topic being discussed)
    - Link the aspect to the MAIN topic, NOT to intermediate responses
    - Examples:
      - Conversation: jaundice → cure of jaundice → user says "prevention" → reformulate to "prevention of jaundice" (NOT "prevention of cure of jaundice")
      - Conversation: diabetes → treatment of diabetes → user says "symptoms" → reformulate to "symptoms of diabetes" (NOT "symptoms of treatment of diabetes")
    - Common aspects: prevention, cure, treatment, symptoms, causes, types, diagnosis, complications, management
  
  * For SHORT QUESTIONS (what, how, when, why):
    - If related to the MAIN topic, reformulate as "[question] about [MAIN topic]"
    - Example: Conversation about jaundice, user says "how to treat?" → "How to treat jaundice?"
  
  * Always expand short follow-ups into full standalone questions that clearly specify the topic
- Correct spelling/abbreviations when meaning changes
- Expand medical abbreviations (IUFD -> Intrauterine fetal death)
- Keep rewritten_query concise and medically precise

CRITICAL FOR FOLLOW_UP: 
- When chat_history contains multiple Q&A pairs, identify the MAIN topic from the conversation flow
- Link aspect-based follow-ups (prevention, cure, treatment, etc.) to the MAIN topic, not intermediate responses
- Only use the most recent response for direct affirmations ("yes", "sure") to specific offers

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
-> {{"classification":"MEDICAL_QUESTION","reason":"asking about newborn care","rewritten_query":"Give information about newborn feeding and care","corrections":{{}},"sample_answer":"Newborn feeding involves several key considerations. Breastfeeding is recommended as the primary method, providing essential nutrients and antibodies. Newborns typically feed every 2-3 hours, with feeding sessions lasting 15-20 minutes per breast. Proper latch is crucial to ensure effective milk transfer and prevent nipple soreness. Signs of adequate feeding include regular weight gain, 6-8 wet diapers daily, and contentment after feeds. For formula feeding, use appropriate amounts based on age and follow sterilization guidelines. Burping after feeds helps prevent gas and discomfort."}}

chat_history: ""
question: "hi, how are you?"
-> {{"classification":"CHITCHAT","reason":"greeting","rewritten_query":"hi, how are you?","corrections":{{}}}}

chat_history: ""
question: "when to bath my new born?"
-> {{"classification":"MEDICAL_QUESTION","reason":"asking about newborn care","rewritten_query":"Give information about newborn bathing and care","corrections":{{"bath":"bathe"}}}}

chat_history: ""
question: "what causes jaundice?"
-> {{"classification":"MEDICAL_QUESTION","reason":"standalone medical question","rewritten_query":"Give information about jaundice causes and treatment","corrections":{{}},"sample_answer":"Jaundice is caused by an excess of bilirubin in the blood. Common causes include liver diseases such as hepatitis, cirrhosis, or liver damage. Hemolytic anemia, where red blood cells break down rapidly, can also lead to jaundice. In newborns, physiological jaundice is common due to immature liver function. Blocked bile ducts from gallstones or tumors can prevent bile excretion, causing jaundice. Certain medications and genetic conditions like Gilbert syndrome may also contribute to elevated bilirubin levels."}}

chat_history: ""
question: "baby is not sucking mother's milk"
-> {{"classification":"MEDICAL_QUESTION","reason":"asking about newborn feeding issues","rewritten_query":"Give information about newborn feeding problems and solutions","corrections":{{}},"sample_answer":"Newborn feeding problems can arise from various causes. Latch issues are common and may result from improper positioning or tongue-tie. Weak sucking reflex can occur in premature infants or those with neurological conditions. Fatigue or jaundice may reduce feeding interest. Solutions include ensuring proper positioning, consulting lactation specialists, checking for oral abnormalities, and monitoring hydration. In some cases, supplemental feeding or pumping may be necessary. Persistent issues require medical evaluation to rule out underlying conditions affecting feeding ability."}}

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

chat_history: "Assistant: Would you like to know about newborn feeding patterns?"
question: "yes"
-> {{"classification":"FOLLOW_UP","reason":"affirmation to most recent assistant suggestion about newborn feeding patterns","rewritten_query":"Provide details on newborn feeding patterns","corrections":{{}}}}

NOTE: Even if the assistant's previous response was a "no context" message that included a follow-up offer like "Would you like to know about X?", the chat_history will contain "Assistant: Would you like to know about X?" and you should use THIS for FOLLOW_UP reformulation.

EXAMPLES OF CONTEXT-AWARE REFORMULATION:

chat_history: 
User: jaundice | Assistant: [provides information about jaundice]
User: cure | Assistant: [provides information about cure of jaundice]
question: "prevention"
-> {{"classification":"FOLLOW_UP","reason":"follow-up asking about prevention, linking to main topic jaundice","rewritten_query":"Give information about prevention of jaundice","corrections":{{}}}}

chat_history: 
User: diabetes | Assistant: [provides information about diabetes]
User: treatment | Assistant: [provides information about treatment of diabetes]
question: "symptoms"
-> {{"classification":"FOLLOW_UP","reason":"follow-up asking about symptoms, linking to main topic diabetes","rewritten_query":"Give information about symptoms of diabetes","corrections":{{}}}}

chat_history: 
User: jaundice | Assistant: [provides information about jaundice]
User: cure | Assistant: Would you like to know about complications of jaundice?
question: "yes"
-> {{"classification":"FOLLOW_UP","reason":"direct affirmation to most recent offer","rewritten_query":"Provide details on complications of jaundice","corrections":{{}}}}

chat_history: 
User: jaundice | Assistant: [provides information about jaundice]
User: cure | Assistant: [no context response] Would you like to know about newborn feeding patterns?
question: "yes"
-> {{"classification":"FOLLOW_UP","reason":"direct affirmation to most recent offer about newborn feeding","rewritten_query":"Provide details on newborn feeding patterns","corrections":{{}}}}

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

TASK OVERVIEW:
1. First, evaluate each context chunk to determine if it can answer the query
2. Use only qualified chunks that contain relevant information
3. Consolidate information from multiple sources intelligently
4. Generate a concise, well-structured answer

CRITICAL RULES:
- Always answer ONLY from <context> provided below which caters to the query
- NEVER use the web, external sources, or prior knowledge outside the given context
- Be Very PRECISE AND TO THE POINT ABOUT WHAT IS ASKED IN THE QUERY
- ANSWER STRICTLY IN 150-200 WORDS (USING BULLET AND SUB-BULLET POINTS [MAX 5-6 POINTS])
- Each bullet must be factual and context-bound
- Summarize meaningfully in a perfect flow
- Do NOT mention references to what's not-there in the context (e.g., "the document doesn't have much info about X" ❌)

STEP 1 - FILTERING:
- Evaluate each context chunk individually
- Use only chunks that contain information relevant to answering the query
- Ignore chunks that don't address the query topic
- If a chunk has partial but relevant information, include it

STEP 2 - MULTI-SOURCE CONSOLIDATION (CRITICAL):
When multiple sources contain information about the same topic:

DO (CORRECT):
✅ "According to A.pdf, B.pdf, C.pdf:
   • Point 1: [Combined information from all sources]
   • Point 2: [Unique fact from A and B]
   • Point 3: [Unique fact from C only]"

DON'T (WRONG):
❌ "According to A.pdf:
   • Point 1: [facts from A]
   According to B.pdf:
   • Point 2: [facts from B]
   According to C.pdf:
   • Point 3: [facts from C]"

CONSOLIDATION RULES:
- Group sources that contribute to the same point: "According to A.pdf, B.pdf"
- If sources provide unique information, combine them: "According to A.pdf, B.pdf, C.pdf"
- Extract unique facts from each source and merge them logically
- Avoid repetition: If A and B say the same thing, mention it once
- Maintain flow: Related facts should be grouped together, not separated by source
- Extract filename from metadata (DON'T mention extensions - abc✅, abc.pdf❌)

STEP 3 - ANSWER FORMATTING:
- Begin with: **"According to <source(s)>"** (extract filename, NO extensions)
  - Single source: "According to abc"
  - Multiple sources: "According to abc, def, ghi"
- BE VERY PRECISE AND CONCISE IN ANSWER FRAMING STRUICTLY ANSWER WHAT HAS BEEN ASKED (STRICTLY FROM THE CONTEXT PROVIDED AND NOT FROM OPEN WEB) IN MAXIUMUM OF 150-200 WORDS (SO CHOOSE WISELY) AND AVOID PROVIDING IRRELEVANT INFORMATION 
- Use bullet and sub-bullet points (MAXIMUM 5 POINTS MINIMUM)
- Each bullet should be a complete, meaningful point
- Combine related information from multiple sources into single bullets
- Ensure logical flow: Related points should be adjacent


STEP 4 - FOLLOW-UP QUESTION (MANDATORY IF context_followup PROVIDED):
- If <context_followup> is non-empty, you MUST include a follow-up question
- Format: "Would you like to know about <topic from context_followup>?" (without bullet points, at the end of answer)
- Follow-up should be about a medical topic (not general/document titles like "Clinical Reference Manual for Advanced Neonatal Care in Ethiopia" ❌)
- Follow-up should NOT overlap with information already in the answer
- The follow-up must be STRICTLY from context_followup

CRITICAL FOR FOLLOW-UP GENERATION:
- Read the ENTIRE context_followup content carefully
- Extract the MAIN MEDICAL TOPIC or KEY TERM from context_followup
- Look for medical conditions, treatments, symptoms, complications, or procedures mentioned
- Generate a question that will match relevant medical documents when searched
- Use specific medical terminology that appears in medical documents
- Avoid generic phrases - use precise medical terms (e.g., "treatment for jaundice" not "more information")
- Ensure the question format matches how medical documents are indexed (e.g., "What are the complications of X?", "How to manage Y?", "What are the symptoms of Z?")
- The question should be searchable and will retrieve relevant chunks from the document database
- If context_followup contains multiple topics, pick the most relevant/interesting one that doesn't overlap with your answer

EXAMPLES:

Example 1 - Single Source:
Query: "What causes fever?"
Context:
From [medical_guide.pdf]:
Fever is caused by infections, inflammatory conditions, and certain medications.

Answer:
According to medical_guide:
• Fever is primarily caused by infections such as bacterial or viral illnesses
• Inflammatory conditions can trigger fever responses
• Certain medications may induce fever as a side effect

---

Example 2 - Multi-Source Consolidation (CORRECT):
Query: "What are the symptoms of jaundice?"
Context:
From [symptom_guide.pdf]:
Jaundice symptoms include yellowing of skin, yellowing of eyes, dark urine.

From [clinical_manual.pdf]:
Jaundice presents with yellow skin discoloration, pale stools, and fatigue.

From [treatment_guide.pdf]:
Jaundice symptoms: yellowing, itchy skin, and abdominal pain.

Answer:
According to symptom_guide, clinical_manual, treatment_guide:
• Yellowing of skin and eyes (sclera) is the primary visible symptom
• Dark urine and pale stools indicate bile pigment changes
• Fatigue and abdominal pain are common accompanying symptoms
• Itchy skin (pruritus) may occur in some cases

---
Example 4 - Precise Answer Formatting:
Query: "Oxytocin dosage"

Expected answer : * The recommended dosage for postpartum hemorrhage management is 10 IU by IM/IV bolus injection, administered slowly over 1-2 minutes if IV access is available. ✅
                  * For induction or augmentation of labor, oxytocin can be infused at a concentration of 2.5 units in 500 mL of dextrose or normal saline, starting at 2.5 mIU per minute and increasing by 2.5 mIU per minute every 30 minutes until a good contraction pattern is established. ✅

Not Expected answer : * Oxytocin is an injectable uterotonic drug used to cause uterine contractions, assisting in the separation of the placenta and stopping postpartum bleeding.. ❌
                     *  It is recommended for actively managing the third stage of labor and should be given within one minute of the birth of the last baby. ❌

Example 3 - Multi-Source Consolidation (WRONG - DON'T DO THIS):
Query: "What are the symptoms of jaundice?"
Answer:
According to symptom_guide:
• Yellowing of skin
• Yellowing of eyes
• Dark urine
According to clinical_manual:
• Yellow skin discoloration
• Pale stools
• Fatigue
According to treatment_guide:
• Yellowing
• Itchy skin
• Abdominal pain

❌ This is WRONG because:
- Repetitive source citations break flow
- Same information repeated multiple times
- Poor readability and structure
- Wastes word count

---

Example 4 - Partial Information from Multiple Sources:
Query: "How to treat dehydration?"
Context:
From [hydration_guide.pdf]:
Oral rehydration solutions are effective. Drink fluids regularly.

From [emergency_care.pdf]:
Severe dehydration requires IV fluids and medical supervision.

Answer:
According to hydration_guide, emergency_care:
• Mild to moderate dehydration can be treated with oral rehydration solutions
• Regular fluid intake is essential for recovery
• Severe cases require IV fluids and immediate medical supervision
• Medical monitoring is crucial for severe dehydration cases

---

Example 5 - Follow-up Question Generation:
Query: "What causes jaundice?"
Context:
From [jaundice_guide.pdf]:
Jaundice is caused by bilirubin buildup...

Followup Context:
From [complications_guide.pdf]:
Complications of jaundice include liver damage, kernicterus in newborns, and chronic liver disease.

Answer:
According to jaundice_guide:
• Jaundice is caused by bilirubin buildup in the blood
• Common causes include liver diseases and hemolytic anemia
• Newborn jaundice is common due to immature liver function

Would you like to know about complications of jaundice?

Note: The follow-up question extracted the main medical topic "complications of jaundice" from the followup_context, which is a specific, searchable medical term that will match documents.

---

WORD LIMIT AND STRUCTURE:
- STRICTLY 150-200 words total
- Use 4-5 bullet points if sufficient information is available
- Minimum 3 bullet points required
- Each bullet should be substantive, not trivial
- Maintain professional medical expert level English
- Ensure perfect flow of information (not random spitting)

QUALITY CHECKS:
Before finalizing, ensure:
- ✅ Multiple sources consolidated into single citation when appropriate
- ✅ No repetition of same information
- ✅ Logical grouping of related facts
- ✅ Smooth flow between points
- ✅ All information is from context only
- ✅ Follow-up question doesn't overlap with answer
- ✅ Follow-up question uses specific medical terminology that will match documents
- ✅ Word count within 150-200 words
- ✅ Sources listed without file extensions

FOLLOW-UP QUESTION EXAMPLES:

GOOD (Will match documents):
✅ "Would you like to know about jaundice treatment options?"
✅ "Would you like to know about complications of postpartum hemorrhage?"
✅ "Would you like to know about newborn feeding techniques?"
✅ "Would you like to know about prevention of preeclampsia?"

BAD (Won't match documents well):
❌ "Would you like to know more about this topic?"
❌ "Would you like to know about Clinical Reference Manual?"
❌ "Would you like additional information?"
❌ "Would you like to know about the document?"

<user_query>
{query}
</user_query>

<context>
{context}
</context>

<followup_context>
{context_followup}
</followup_context>

INSTRUCTIONS:
1. First, evaluate each chunk in <context> to determine if it can answer the query
2. Filter out chunks that don't address the query - if chunks seem irrelevant, still try to extract any useful information
3. Identify which sources have relevant information
4. Consolidate information from multiple sources (use format: "According to A, B, C" when multiple sources contribute)
5. Generate answer following all rules above
6. MANDATORY: If <followup_context> is provided and non-empty, you MUST include a follow-up question at the end
   - Read the entire <followup_context> content
   - Extract the MAIN MEDICAL TOPIC or KEY TERM from <followup_context>
   - Generate: "Would you like to know about [specific medical topic]?"
   - Use precise medical terminology that will match document searches
   - Example: "Would you like to know about jaundice treatment?" (not "Would you like more information?")
   - The follow-up question should be the LAST line of your response

IMPORTANT: If chunks in <context> seem partially relevant or borderline, still use them - your role is to judge and extract useful information. Only exclude chunks that are completely unrelated to the query.

REMINDER: If <followup_context> is provided, you MUST generate a follow-up question. Do not skip this step.

Write the final answer now.
"""
)

# -------------------- DOCUMENT ABBREVIATION MAPPING --------------------
# Mapping from abbreviations to full document names (case-insensitive lookup)
# All keys are normalized to lowercase for lookup
DOCUMENT_ABBREVIATION_MAP = {
    # BABC - Bleeding After Birth Provider Guide
    "babc": "Bleeding After Birth Provider Guide",
    "babc.pdf": "Bleeding After Birth Provider Guide",
    "babg": "Bleeding After Birth Provider Guide",  # Alternative spelling
    "babg.pdf": "Bleeding After Birth Provider Guide",
    # ECLB - Essential Care for Labor and Birth Provider Guide
    "eclb": "Essential Care for Labor and Birth Provider Guide",
    "eclb.pdf": "Essential Care for Labor and Birth Provider Guide",
    "enc": "Essential Newborn Care: Immediate Care and Helping Babies Breathe at Birth Provider Guide",
    "enc.pdf": "Essential Newborn Care: Immediate Care and Helping Babies Breathe at Birth Provider Guide",
    "enc 1": "Essential Newborn Care: Immediate Care and Helping Babies Breathe at Birth Provider Guide",
    "enc1": "Essential Newborn Care: Immediate Care and Helping Babies Breathe at Birth Provider Guide",
    "enc 1.pdf": "Essential Newborn Care: Immediate Care and Helping Babies Breathe at Birth Provider Guide",
    "enc1.pdf": "Essential Newborn Care: Immediate Care and Helping Babies Breathe at Birth Provider Guide",
    "enc 2": "Essential Newborn Care: Assessment and Continuing Care Provider Guide",
    "enc2": "Essential Newborn Care: Assessment and Continuing Care Provider Guide",
    "enc 2.pdf": "Essential Newborn Care: Assessment and Continuing Care Provider Guide",
    "enc2.pdf": "Essential Newborn Care: Assessment and Continuing Care Provider Guide",
    "pee": "Pre-eclampsia & Eclampsia Provider Guide",
    "29_jan_morning": "WHO",
    "pee.pdf": "Pre-eclampsia & Eclampsia Provider Guide",
    "pee 1": "HMBS: Pre-Eclampsia & Eclampsia Medication Table",
    "pee1": "HMBS: Pre-Eclampsia & Eclampsia Medication Table",
    "pee 1.pdf": "HMBS: Pre-Eclampsia & Eclampsia Medication Table",
    "pee1.pdf": "HMBS: Pre-Eclampsia & Eclampsia Medication Table",
    "pee 2": "HMBS: MgSO4 Dosing and Monitoring Checklist",
    "pee2": "HMBS: MgSO4 Dosing and Monitoring Checklist",
    "pee 2.pdf": "HMBS: MgSO4 Dosing and Monitoring Checklist",
    "pee2.pdf": "HMBS: MgSO4 Dosing and Monitoring Checklist",
    "pol": "Prolonged & Obstructed Labor: Assessment Provider Guide",
    "pol.pdf": "Prolonged & Obstructed Labor: Assessment Provider Guide",
    "pol 1": "Prolonged & Obstructed Labor: Assessment Provider Guide",
    "pol1": "Prolonged & Obstructed Labor: Assessment Provider Guide",
    "pol 1.pdf": "Prolonged & Obstructed Labor: Assessment Provider Guide",
    "pol1.pdf": "Prolonged & Obstructed Labor: Assessment Provider Guide",
    "pol 2": "Prolonged & Obstructed Labor: Management Provider Guide",
    "pol2": "Prolonged & Obstructed Labor: Management Provider Guide",
    "pol 2.pdf": "Prolonged & Obstructed Labor: Management Provider Guide",
    "pol2.pdf": "Prolonged & Obstructed Labor: Management Provider Guide",
    "vab": "Vacuum Assisted Birth Provider Guide",
    "vab.pdf": "Vacuum Assisted Birth Provider Guide",
    "who_newborn_health": "Newborn Health Approved by WHO",
    "who_newborn_health.pdf": "Newborn Health Approved by WHO",
    "who_pregnancy_birth": "Managing Complications in Pregnancy and Childbirth by WHO",
    "who_pregnancy_birth.pdf": "Managing Complications in Pregnancy and Childbirth by WHO",
}

def expand_document_name(abbrev: str) -> str:
    """
    Expand document abbreviation to full name.
    Handles edge cases: case-insensitive, with/without spaces, with/without extensions, double extensions.
    
    Args:
        abbrev: Document abbreviation (e.g., "pee", "PEE", "pee.pdf", "eclb.pdf.pdf", "enc 1")
    
    Returns:
        Full document name if found, otherwise returns original abbrev
    """
    if not abbrev or abbrev == "unknown":
        return abbrev
    
    # Normalize: lowercase, strip whitespace
    normalized = abbrev.lower().strip()
    
    # Remove all .pdf extensions (handles "file.pdf.pdf" edge case)
    while normalized.endswith('.pdf'):
        normalized = normalized[:-4]
    
    # Remove any remaining file extensions just in case
    if '.' in normalized:
        normalized = normalized.rsplit('.', 1)[0]
    
    # Normalize spaces and underscores: "enc 1" -> "enc1", "enc  1" -> "enc1", "enc_1" -> "enc1"
    normalized_no_spaces = normalized.replace(" ", "").replace("_", "")
    
    # Try exact match first (normalized key)
    if normalized_no_spaces in DOCUMENT_ABBREVIATION_MAP:
        return DOCUMENT_ABBREVIATION_MAP[normalized_no_spaces]
    
    # Also try with original spaces/underscores preserved (for keys like "who_newborn_health")
    if normalized in DOCUMENT_ABBREVIATION_MAP:
        return DOCUMENT_ABBREVIATION_MAP[normalized]
    
    # Handle numbered variants (e.g., "enc1", "enc 1", "pee2")
    if len(normalized_no_spaces) >= 3 and normalized_no_spaces[-1].isdigit():
        base = normalized_no_spaces[:-1]
        number = normalized_no_spaces[-1]
        
        # Try various formats: "enc1", "enc 1", "enc_1"
        variants = [
            f"{base}{number}",      # "enc1"
            f"{base} {number}",      # "enc 1" (with space)
            f"{base}_{number}",      # "enc_1" (with underscore)
        ]
        
        for variant in variants:
            if variant in DOCUMENT_ABBREVIATION_MAP:
                return DOCUMENT_ABBREVIATION_MAP[variant]
        
        # Also try base name without number (for generic "enc" when "enc1" not found)
        if base in DOCUMENT_ABBREVIATION_MAP:
            return DOCUMENT_ABBREVIATION_MAP[base]
    
    # Try normalized_no_spaces variants for numbered cases too
    if len(normalized) >= 3 and normalized[-1].isdigit():
        base = normalized[:-1]
        number = normalized[-1]
        variants = [
            f"{base}{number}",
            f"{base} {number}",
            f"{base}_{number}",
        ]
        for variant in variants:
            if variant in DOCUMENT_ABBREVIATION_MAP:
                return DOCUMENT_ABBREVIATION_MAP[variant]
        if base in DOCUMENT_ABBREVIATION_MAP:
            return DOCUMENT_ABBREVIATION_MAP[base]
    
    # Fallback: return original (might be a new document not in mapping)
    return abbrev

def expand_sources_string(sources_list: List[str]) -> str:
    """
    Expand all document abbreviations in sources list to full names.
    Handles edge cases: multiple extensions, case variations, spaces.
    
    Args:
        sources_list: List of source abbreviations (e.g., ["pee.pdf", "eclb.pdf.pdf", "enc 1"])
    
    Returns:
        String with full names joined by " and " (e.g., "Pre-eclampsia & Eclampsia Provider Guide and Essential Care for Labor and Birth Provider Guide")
    """
    if not sources_list:
        return "unknown"
    
    expanded = []
    for src in sources_list:
        if not src or src == "unknown":
            expanded.append("unknown")
            continue
        
        # expand_document_name handles all normalization internally (extensions, case, spaces)
        # Pass the full source name - function handles .pdf removal
        full_name = expand_document_name(src)
        expanded.append(full_name)
        print(f"[expand_sources] '{src}' -> '{full_name}'")
    
    result = " and ".join(expanded)
    return result

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
    
    # 2) CHROMADB VECTOR SEARCH
    print("[HYBRID] Step 2: Querying ChromaDB vector database...")
    if chromadb_collection is None:
        raise RuntimeError("ChromaDB collection not initialized. Run setup_local_vectorstore.py first.")
    
    # Check collection has data
    collection_count = chromadb_collection.count()
    if collection_count == 0:
        print(f"[HYBRID] WARNING: ChromaDB collection is empty ({collection_count} chunks)")
        print("[HYBRID] Returning empty results - run setup_local_vectorstore.py to populate the collection")
        timer.mark("chromadb.query")
        print(f"[HYBRID] chromadb.query took {timer.last - timer.start:.3f}s")
        matches = []
        print(f"[HYBRID] Retrieved {len(matches)} matches from ChromaDB")
    else:
        print(f"[HYBRID] Querying ChromaDB collection with {collection_count} chunks...")
        try:
            # ChromaDB query with error handling
            results = chromadb_collection.query(
                query_embeddings=[q_emb],
                n_results=min(top_k_vec, collection_count),  # Don't request more than available
                include=["metadatas", "documents", "distances"]
            )
            timer.mark("chromadb.query")
            print(f"[HYBRID] chromadb.query took {timer.last - timer.start:.3f}s")
            
            # Process results (ChromaDB returns lists of lists)
            matches = []
            if results and "ids" in results and results["ids"] and len(results["ids"]) > 0:
                num_results = len(results["ids"][0])
                for i in range(num_results):
                    try:
                        match_id = results["ids"][0][i]
                        metadata = results["metadatas"][0][i] if (results.get("metadatas") and len(results["metadatas"]) > 0) else {}
                        document = results["documents"][0][i] if (results.get("documents") and len(results["documents"]) > 0) else ""
                        distance = results["distances"][0][i] if (results.get("distances") and len(results["distances"]) > 0) else 1.0
                        
                        # Convert distance to similarity score (ChromaDB uses distance, lower is better)
                        # For cosine similarity: distance = 1 - cosine_similarity
                        # So: similarity = 1 - distance (for cosine space)
                        # Cosine distance range: 0 (identical) to 2 (opposite)
                        # Convert to similarity: similarity = 1 - (distance / 2)
                        if distance is not None:
                            # For cosine, ChromaDB returns 1 - cosine_similarity as distance
                            # So similarity = 1 - distance
                            similarity = max(0.0, 1.0 - float(distance)) if distance <= 2.0 else 0.0
                        else:
                            similarity = 0.0
                        
                        matches.append({
                            "id": match_id,
                            "metadata": metadata if isinstance(metadata, dict) else {},
                            "document": document if document else "",
                            "score": similarity
                        })
                    except Exception as e:
                        print(f"[HYBRID] WARNING: Error processing match {i}: {e}")
                        continue
            else:
                print(f"[HYBRID] WARNING: No results returned from ChromaDB query")
                matches = []
            
            print(f"[HYBRID] Retrieved {len(matches)} matches from ChromaDB")
        except Exception as e:
            print(f"[HYBRID] ERROR: ChromaDB query failed: {e}")
            timer.mark("chromadb.query")
            print(f"[HYBRID] chromadb.query took {timer.last - timer.start:.3f}s (failed)")
            raise RuntimeError(f"ChromaDB query failed: {e}") from e

    # 3) PROCESS MATCHES
    print("[HYBRID] Step 3: Processing matches and deduplicating...")
    vec_matches = []
    for m in matches:
        try:
            # Get text from metadata first, then fallback to document
            text = m["metadata"].get("text_full") or m["metadata"].get("text_snippet") or m.get("document", "")
            if not text:
                print(f"[HYBRID] WARNING: Match {m['id']} has no text content, skipping")
                continue
            
            vec_matches.append({
                "id": m["id"],
                "text": text,
                "meta": m["metadata"]
            })
        except Exception as e:
            print(f"[HYBRID] WARNING: Error processing match {m.get('id', 'unknown')}: {e}")
            continue
    
    timer.mark("process_matches")
    print(f"[HYBRID] process_matches took {timer.last - timer.start:.3f}s")

    # 4) COMBINE & CAP
    if not vec_matches:
        print(f"[HYBRID] WARNING: No valid matches after processing")
        candidates = []
    else:
        candidates = {m["id"]: m for m in (vec_matches)}  # deduplicate
        candidates = list(candidates.values())[:u_cap]
    timer.mark("deduplicate_cap")
    print(f"[HYBRID] deduplicate_cap took {timer.last - timer.start:.3f}s")
    print(f"[HYBRID] Final candidates: {len(candidates)}")

    # 5) RE-RANKING
    print("[HYBRID] Step 4: Re-ranking candidates...")
    if not candidates:
        print(f"[HYBRID] WARNING: No candidates to re-rank, returning empty list")
        return []
    
    # Process re-ranking (we know candidates exist here)
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

# -------------------- BERT FILTER + ANSWER --------------------
def bert_filter_candidates(candidates: List[Dict[str, Any]], sample_answer: str, original_query: str, min_score: float = 0.35) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filter candidates using cross-encoder scores (BERT-based filtering):
    - Uses sample_answer for comparison if available (better semantic matching)
    - Falls back to original_query if sample_answer is empty
    - Top 4 candidates with score > min_score for answer
    - Next best 2 candidates for followup
    
    Args:
        candidates: List of candidate chunks with 'scores' dict containing 'cross' score
        sample_answer: Sample answer text from reformulation LLM (for better semantic matching)
        original_query: Original query (fallback if sample_answer is empty)
        min_score: Minimum cross-encoder score threshold (default: 0.35)
    
    Returns:
        Dict with 'answer_chunks' and 'followup_chunks'
    """
    print(f"[BERT_FILTER] Filtering {len(candidates)} candidates with min_score={min_score}")
    print(f"[BERT_FILTER] Using {'sample_answer' if sample_answer else 'original_query'} for comparison")
    timer = CheckpointTimer("bert_filter_candidates")
    
    # Use sample_answer for comparison if available, otherwise fall back to original_query
    comparison_text = sample_answer if sample_answer else original_query
    
    # Re-score candidates against sample_answer using cross-encoder
    global reranker
    print(f"[BERT_FILTER] Re-scoring candidates against comparison text ({len(comparison_text)} chars)...")
    print(f"[BERT_FILTER] Comparison text preview: {comparison_text[:200]}...")
    print(f"[BERT_FILTER] Reranker type: {type(reranker).__name__}")
    print(f"[BERT_FILTER] Reranker has 'predict': {hasattr(reranker, 'predict')}")
    print(f"[BERT_FILTER] Reranker has 'rerank': {hasattr(reranker, 'rerank')}")
    print(f"[BERT_FILTER] FLASHRANK_AVAILABLE: {FLASHRANK_AVAILABLE}")
    
    # Debug: Check candidate texts
    print(f"[BERT_FILTER] Checking first candidate text length: {len(candidates[0].get('text', '')) if candidates else 0}")
    if candidates:
        print(f"[BERT_FILTER] First candidate text preview: {candidates[0].get('text', '')[:100]}...")
        print(f"[BERT_FILTER] First candidate original cross score: {candidates[0].get('scores', {}).get('cross', 0.0)}")
    
    # Batch re-scoring for efficiency
    # IMPORTANT: Check which reranker is actually being used
    # FlashRank (Ranker) has 'rerank' method
    # CrossEncoder has 'predict' method
    try:
        # Priority: Check FlashRank first if available, then CrossEncoder
        if hasattr(reranker, "rerank") and FLASHRANK_AVAILABLE:
            # FlashRank path - batch rerank
            # NOTE: FlashRank might not work well with very long queries (sample_answer)
            # Consider truncating sample_answer or using rewritten query for FlashRank
            print("[BERT_FILTER] Using FlashRank.rerank path")
            
            # For FlashRank, use original_query if sample_answer is too long (FlashRank optimized for queries, not long answers)
            flashrank_query = original_query if len(comparison_text) > 200 else comparison_text
            print(f"[BERT_FILTER] Using {'original_query' if len(comparison_text) > 200 else 'sample_answer'} for FlashRank (len={len(flashrank_query)})")
            
            passages = []
            id_to_index = {}
            for i, candidate in enumerate(candidates):
                cand_text = candidate.get("text", "")
                if cand_text and len(cand_text.strip()) > 0:
                    cand_id = candidate.get("id", "")
                    if not cand_id:
                        cand_id = f"cand_{i}"
                    passages.append({"id": cand_id, "text": cand_text})
                    id_to_index[cand_id] = i
                else:
                    print(f"[BERT_FILTER] WARNING: Candidate {i} has empty text, skipping")
            
            if not passages:
                print("[BERT_FILTER] ERROR: No valid passages found!")
                for candidate in candidates:
                    candidate["scores"]["answer_match"] = candidate.get("scores", {}).get("cross", 0.0)
            else:
                print(f"[BERT_FILTER] Created {len(passages)} valid passages for reranking")
                print(f"[BERT_FILTER] First passage preview: id={passages[0].get('id')}, text_len={len(passages[0].get('text', ''))}")
                
                request = RerankRequest(query=flashrank_query, passages=passages)
                ranked = reranker.rerank(request)
                print(f"[BERT_FILTER] FlashRank returned {len(ranked) if ranked else 0} ranked results")
                
                if ranked:
                    print(f"[BERT_FILTER] First ranked result full structure: {ranked[0]}")
                    print(f"[BERT_FILTER] First ranked result keys: {ranked[0].keys() if isinstance(ranked[0], dict) else 'Not a dict'}")
                
                # FlashRank returns: {"id": "...", "score": float} or {"id": "...", "relevance_score": float}
                # Check which field name is used
                score_field = None
                if ranked and len(ranked) > 0:
                    first_result = ranked[0]
                    if isinstance(first_result, dict):
                        if "score" in first_result:
                            score_field = "score"
                        elif "relevance_score" in first_result:
                            score_field = "relevance_score"
                
                print(f"[BERT_FILTER] FlashRank score field: {score_field}")
                
                # Create a mapping of id to score
                score_map = {}
                if ranked and score_field:
                    for r in ranked:
                        r_id = r.get("id")
                        r_score = r.get(score_field, 0.0)
                        if r_id:
                            score_map[r_id] = float(r_score)
                            print(f"[BERT_FILTER] Mapped id={r_id[:20]}... -> score={float(r_score):.4f}")
                
                print(f"[BERT_FILTER] Score map size: {len(score_map)}, keys: {list(score_map.keys())[:3] if score_map else []}")
                
                # Map scores back to candidates
                for i, candidate in enumerate(candidates):
                    cand_id = candidate.get("id", "")
                    if not cand_id:
                        cand_id = f"cand_{i}"
                    
                    # Try exact ID match
                    score_val = score_map.get(cand_id)
                    
                    if score_val is None:
                        # Try fallback ID format
                        fallback_id = f"cand_{i}"
                        score_val = score_map.get(fallback_id)
                    
                    if score_val is None:
                        # Use original cross score as fallback
                        score_val = candidate.get("scores", {}).get("cross", 0.0)
                        print(f"[BERT_FILTER] Candidate {i}: No score found (id={cand_id[:20]}...), using original cross score={score_val:.4f}")
                    else:
                        print(f"[BERT_FILTER] Candidate {i}: id={cand_id[:20]}..., score={score_val:.4f}")
                    
                    candidate["scores"]["answer_match"] = score_val
                    
        elif hasattr(reranker, "predict"):
            # CrossEncoder path - batch predict
            # NOTE: CrossEncoder expects (query, text) pairs
            print("[BERT_FILTER] Using CrossEncoder.predict path")
            
            # Check if candidate texts are not empty
            valid_candidates = []
            valid_indices = []
            for i, candidate in enumerate(candidates):
                cand_text = candidate.get("text", "")
                if cand_text and len(cand_text.strip()) > 0:
                    valid_candidates.append((comparison_text, cand_text))
                    valid_indices.append(i)
                else:
                    print(f"[BERT_FILTER] WARNING: Candidate {i} has empty text, skipping")
            
            if not valid_candidates:
                print("[BERT_FILTER] ERROR: No valid candidates with text found!")
                for candidate in candidates:
                    candidate["scores"]["answer_match"] = candidate.get("scores", {}).get("cross", 0.0)
            else:
                print(f"[BERT_FILTER] Created {len(valid_candidates)} valid pairs for prediction")
                print(f"[BERT_FILTER] First pair preview: query_len={len(valid_candidates[0][0])}, text_len={len(valid_candidates[0][1])}")
                
                # Test with a single pair first to verify reranker works
                if len(valid_candidates) > 0:
                    test_pair = valid_candidates[0]
                    print(f"[BERT_FILTER] Testing reranker with single pair: query_len={len(test_pair[0])}, text_len={len(test_pair[1])}")
                    try:
                        test_score = reranker.predict([test_pair])
                        print(f"[BERT_FILTER] Test score result: {test_score}, type: {type(test_score)}")
                        if hasattr(test_score, '__len__') and len(test_score) > 0:
                            test_val = float(test_score[0])
                            print(f"[BERT_FILTER] Test score value: {test_val:.4f}")
                            if test_val == 0.0:
                                print("[BERT_FILTER] WARNING: Test score is 0.0 - reranker might not be working correctly!")
                    except Exception as test_e:
                        print(f"[BERT_FILTER] ERROR in test prediction: {test_e}")
                        import traceback
                        traceback.print_exc()
                
                scores = reranker.predict(valid_candidates)
                print(f"[BERT_FILTER] Raw scores from predict: {scores}")
                print(f"[BERT_FILTER] Scores type: {type(scores)}")
                print(f"[BERT_FILTER] Scores length: {len(scores) if hasattr(scores, '__len__') else 'N/A'}")
                
                # Handle numpy array or list
                if hasattr(scores, 'tolist'):
                    scores = scores.tolist()
                    print(f"[BERT_FILTER] Converted numpy array to list")
                elif hasattr(scores, '__iter__') and not isinstance(scores, (str, bytes)):
                    scores = list(scores)
                    print(f"[BERT_FILTER] Converted iterable to list")
                elif not isinstance(scores, list):
                    print(f"[BERT_FILTER] WARNING: Scores is not a list, array, or iterable: {type(scores)}")
                    scores = [scores] if scores is not None else []
                
                print(f"[BERT_FILTER] Processed scores list: {scores}")
                print(f"[BERT_FILTER] Scores range: min={min(scores) if scores else 'N/A'}, max={max(scores) if scores else 'N/A'}")
                
                # Map scores back to original candidates
                score_idx = 0
                for i, candidate in enumerate(candidates):
                    if i in valid_indices:
                        if score_idx < len(scores):
                            try:
                                score_val = float(scores[score_idx])
                                if score_val == 0.0:
                                    print(f"[BERT_FILTER] WARNING: Candidate {i} got score 0.0 from reranker")
                                candidate["scores"]["answer_match"] = score_val
                                print(f"[BERT_FILTER] Candidate {i}: score={score_val:.4f}, id={candidate.get('id', 'unknown')[:50]}")
                                score_idx += 1
                            except (ValueError, TypeError) as conv_e:
                                print(f"[BERT_FILTER] ERROR converting score for candidate {i}: {conv_e}, score={scores[score_idx]}")
                                candidate["scores"]["answer_match"] = candidate.get("scores", {}).get("cross", 0.0)
                                score_idx += 1
                        else:
                            print(f"[BERT_FILTER] Candidate {i}: Score index out of range, using original cross score")
                            candidate["scores"]["answer_match"] = candidate.get("scores", {}).get("cross", 0.0)
                    else:
                        print(f"[BERT_FILTER] Candidate {i}: Empty text, using original cross score")
                        candidate["scores"]["answer_match"] = candidate.get("scores", {}).get("cross", 0.0)
        else:
            # Fallback: use original cross-encoder scores
            print("[BERT_FILTER] No reranker available, using original scores")
            for candidate in candidates:
                candidate["scores"]["answer_match"] = candidate.get("scores", {}).get("cross", 0.0)
    except Exception as e:
        print(f"[BERT_FILTER] Error during batch re-scoring: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to original scores
        for candidate in candidates:
            candidate["scores"]["answer_match"] = candidate.get("scores", {}).get("cross", 0.0)
    
    # Filter by answer_match score threshold
    qualified = [c for c in candidates if c.get("scores", {}).get("answer_match", 0.0) > min_score]
    
    # Sort by answer_match score (descending)
    qualified = sorted(qualified, key=lambda x: x.get("scores", {}).get("answer_match", 0.0), reverse=True)
    
    print(f"[BERT_FILTER] Score distribution: {[round(c.get('scores', {}).get('answer_match', 0.0), 3) for c in candidates[:5]]}")
    
    # FALLBACK: If no qualified chunks, use top 2 based on original cross scores (let LLM judge)
    if len(qualified) == 0:
        print(f"[BERT_FILTER] WARNING: No qualified chunks found (all scores <= {min_score})")
        print(f"[BERT_FILTER] FALLBACK: Using top 2 chunks based on original cross scores for LLM to judge")
        # Sort all candidates by original cross score
        all_sorted = sorted(candidates, key=lambda x: x.get("scores", {}).get("cross", 0.0), reverse=True)
        answer_chunks = all_sorted[:2]
        # For followup, take next best 2 from remaining
        followup_chunks = all_sorted[2:4] if len(all_sorted) > 2 else []
        print(f"[BERT_FILTER] Fallback selected top 2 chunks with cross scores: {[c.get('scores', {}).get('cross', 0.0) for c in answer_chunks]}")
        print(f"[BERT_FILTER] Fallback selected {len(followup_chunks)} followup chunks")
    else:
        # Take top 4 qualified for answer
        answer_chunks = qualified[:4]
        
        # Take next best 2 for followup
        # First try remaining qualified chunks
        remaining = qualified[4:] if len(qualified) > 4 else []
        
        # If we don't have 2 yet, take from unqualified but sorted candidates
        if len(remaining) < 2:
            unqualified = [c for c in candidates if c.get("scores", {}).get("answer_match", 0.0) <= min_score]
            # Sort unqualified by answer_match score (descending), fallback to cross score
            unqualified = sorted(unqualified, key=lambda x: (
                x.get("scores", {}).get("answer_match", 0.0),
                x.get("scores", {}).get("cross", 0.0)
            ), reverse=True)
            remaining = (remaining + unqualified)[:2]
        
        followup_chunks = remaining[:2]
    
    print(f"[BERT_FILTER] Qualified chunks (answer_match score > {min_score}): {len(qualified)}")
    print(f"[BERT_FILTER] Selected {len(answer_chunks)} answer chunks, {len(followup_chunks)} followup chunks")
    
    timer.mark("done")
    return {
        "answer_chunks": answer_chunks,
        "followup_chunks": followup_chunks
    }

# Keep judge_sufficiency for backward compatibility (not used in new flow)
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

    # Build context from top candidates
    sources = []
    ctx_parts = []
    for c in top_candidates[:4]:
        src = c.get("meta", {}).get("doc_name", "unknown")
        if src not in sources:
            sources.append(src)
        # Expand document name in context so LLM sees full names everywhere
        # expand_document_name handles all normalization (extensions, case, spaces)
        src_expanded = expand_document_name(src) if src else "unknown"
        ctx_parts.append(f"From [{src_expanded}]:\\n{c['text']}")
    context = "\\n\\n".join(ctx_parts)
    context_followup_str = context_followup # Renamed to avoid conflict with function parameter
    # Expand document abbreviations to full names for sources string
    sources_str = expand_sources_string(sources) if sources else "unknown"
    print(f"[synthesize_answer] Expanded sources: {sources_str}")

    prompt = answer_prompt.format(
        sources=sources_str,
        context=context,
        context_followup=context_followup_str or "",
        query=query
    )
    print("[answer] sending synthesis prompt to GEMINI LLM (with improved multi-source consolidation)")
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

def classify_and_reformulate(chat_history: str, user_message: str, last_3_qa_pairs: str = "") -> Tuple[str, str, str, Dict[str, str], str]:
    """
    Combined classification and reformulation using Gemini in a single LLM call.
    Returns: (classification, reason, rewritten_query, corrections, sample_answer)
    
    Args:
        chat_history: Full chat history context
        user_message: Current user message
        last_3_qa_pairs: Last 3 non-chitchat Q&A pairs formatted for FOLLOW_UP reformulation
                        Format: "User: <q1> | Bot: <a1>\nUser: <q2> | Bot: <a2>\nUser: <q3> | Bot: <a3>"
                        Allows LLM to understand conversation context and link follow-ups to MAIN topic
    """
    global gemini_llm
    
    print("="*60)
    print("CLASSIFICATION + REFORMULATION BLOCK")
    print("="*60)
    
    # Start timing for this block
    block_start = time.perf_counter()
    
    # Use last_3_qa_pairs if available, otherwise fall back to chat_history
    # This allows LLM to understand conversation flow and link follow-ups to the MAIN topic
    chat_history_for_prompt = last_3_qa_pairs if last_3_qa_pairs else chat_history
    if last_3_qa_pairs:
        print(f"[CLASSIFY_REFORM] Using last 3 Q&A pairs for reformulation context")
        print(f"[CLASSIFY_REFORM] Context preview: {last_3_qa_pairs[:200]}...")
    
    prompt = combined_classify_reformulate_prompt.format(chat_history=chat_history_for_prompt, question=user_message)
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
            sample_answer = parsed.get("sample_answer", "")
            
            # Ensure corrections is always a dict
            if not isinstance(corrections, dict):
                corrections = {}
            
            # For CHITCHAT, use original message unchanged and empty sample_answer
            if classification.upper() == "CHITCHAT":
                print("[CLASSIFY_REFORM] CHITCHAT detected - using original message unchanged")
                rewritten_query = user_message
                corrections = {}
                sample_answer = ""
            
            # Ensure sample_answer is a string (fallback to empty if not provided)
            if not isinstance(sample_answer, str):
                sample_answer = ""
            
            # Print detailed outputs
            print("[CLASSIFY_REFORM] EXTRACTED RESULTS:")
            print(f"  Classification: {classification}")
            print(f"  Reason: {reason}")
            print(f"  Rewritten Query: {rewritten_query}")
            print(f"  Corrections: {corrections}")
            print(f"  Sample Answer Length: {len(sample_answer)} chars")
            if sample_answer:
                print(f"  Sample Answer Full Text: {sample_answer}")
                print(f"  Sample Answer Preview: {sample_answer[:150]}...")
            else:
                print("  Sample Answer: EMPTY (will use original query for BERT comparison)")
            
            block_end = time.perf_counter()
            print(f"[CLASSIFY_REFORM] TOTAL BLOCK TIME: {block_end - block_start:.3f} seconds")
            print("="*60)
            
            return classification, reason, rewritten_query, corrections, sample_answer
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
    return "MEDICAL_QUESTION", "fallback", user_message, {}, ""

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
# Import video embedding cache
try:
    import sys
    from pathlib import Path
    # Add current directory to path for local import
    cache_module_path = Path(__file__).parent
    if str(cache_module_path) not in sys.path:
        sys.path.insert(0, str(cache_module_path))
    from video_embedding_cache import VideoEmbeddingCache
    VIDEO_CACHE_AVAILABLE = True
except ImportError as e:
    VIDEO_CACHE_AVAILABLE = False
    print(f"[VIDEO_SYSTEM] WARNING: video_embedding_cache module not found ({e}), using fallback mode")

class VideoMatchingSystem:
    def __init__(self, video_file_path: str = "./FILES/video_link_topic.xlsx", use_cache: bool = True):
        """Initialize the simplified video matching system using BERT similarity"""
        self.video_file_path = video_file_path
        self.topic_list = []  # List of topic strings
        self.url_list = []    # List of corresponding URLs
        self.topic_embeddings = None  # Pre-computed embeddings
        self.use_cache = use_cache and VIDEO_CACHE_AVAILABLE
        
        # Try to use cache first
        if self.use_cache:
            print("[VIDEO_SYSTEM] Initializing video embedding cache...")
            self.embedding_cache = VideoEmbeddingCache(video_file_path=video_file_path)
            if self.embedding_cache.initialize():
                self.topic_list, self.url_list, self.topic_embeddings = self.embedding_cache.get_cached_embeddings()
                print(f"[VIDEO_SYSTEM] Loaded {len(self.topic_list)} video embeddings from cache")
            else:
                print("[VIDEO_SYSTEM] Cache initialization failed, falling back to standard mode")
                self.use_cache = False
        
        # Fallback: Load video data and initialize model (if cache not available)
        if not self.use_cache:
            self._load_video_data()
            # Initialize BERT model for similarity (only if not using cache)
            print("[VIDEO_SYSTEM] Loading BERT model for semantic similarity...")
            self.similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("[VIDEO_SYSTEM] BERT model loaded successfully")
        else:
            # For cache mode, model will be loaded lazily when encoding answers
            self.similarity_model = None
    
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
        
        # Use cached embeddings if available, otherwise encode on-the-fly
        if self.use_cache and self.topic_embeddings is not None:
            # Only encode the answer (embeddings for topics are pre-computed)
            answer_embedding = self.embedding_cache.encode_answer(answer)
            answer_embedding = answer_embedding.reshape(1, -1)  # Reshape for cosine_similarity
            topic_embeddings = self.topic_embeddings
            print(f"[VIDEO_SYSTEM] Using cached video embeddings ({len(self.topic_list)} videos)")
        else:
            # Fallback: encode answer and all topics (original behavior)
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
        
        # Step 2: Threshold gating without LLM reconfirmation
        if best_similarity >= 0.60:
            video_url = self.url_list[best_idx]
            print(f"[VIDEO_SYSTEM] Similarity >= 0.60, returning video directly: {video_url}")
            return video_url
        else:
            print(f"[VIDEO_SYSTEM] Similarity score {best_similarity:.3f} below threshold 0.60 - no video")
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

# -------------------- CACHE SYSTEM (RERANKER + LLM APPROACH) --------------------
class CacheSystem:
    def __init__(self, cache_file_path: str = "./FILES/cache_questions.xlsx"):
        """Initialize the cache system using reranker (same logic as bert_filter_candidates) + LLM verification"""
        self.cache_file_path = cache_file_path
        self.question_list = []  # List of cached questions
        self.answer_list = []    # List of corresponding answers
        
        # Load cache data
        self._load_cache_data()
        
        # Note: We use global reranker (same as bert_filter_candidates) instead of separate BERT model
    
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
    
    def check_cache(self, sample_answer: str, reformulated_query: str) -> Optional[str]:
        """
        Check cache using BOTH approaches and use the best match:
        1. Compare reformulated_query with cached questions (query-to-question matching)
        2. Compare sample_answer with cached answers (answer-to-answer matching, same logic as bert_filter_candidates)
        
        Uses reranker for both comparisons. Picks the best match from either approach.
        If similarity > 0.6, uses LLM to verify and reframe the cached answer.
        
        Args:
            sample_answer: Sample answer text from reformulation LLM (for better semantic matching)
            reformulated_query: Reformulated query (for query-to-question matching and LLM verification)
        
        Returns:
            Reframed cached answer if match found, None otherwise
        """
        if not self.answer_list or not self.question_list:
            print("[CACHE_SYSTEM] No cache data available")
            return None
        
        global reranker
        if not reranker:
            print("[CACHE_SYSTEM] Reranker not available")
            return None
        
        print("="*60)
        print("CACHE SYSTEM BLOCK")
        print("="*60)
        print(f"[CACHE_SYSTEM] Checking cache using BOTH approaches:")
        print(f"[CACHE_SYSTEM]   1. reformulated_query vs cached_questions (query-to-question)")
        print(f"[CACHE_SYSTEM]   2. sample_answer vs cached_answers (answer-to-answer)")
        print(f"[CACHE_SYSTEM] reformulated_query preview: {reformulated_query[:100]}...")
        print(f"[CACHE_SYSTEM] sample_answer preview: {sample_answer[:200] if sample_answer else '(empty)'}...")
        
        cache_start = time.perf_counter()
        
        try:
            best_match_1_idx = None
            best_match_1_score = 0.0
            approach_1_attempted = False
            best_match_2_idx = None
            best_match_2_score = 0.0
            approach_2_attempted = False
            
            # APPROACH 1: Compare reformulated_query with cached questions
            print("\n[CACHE_SYSTEM] APPROACH 1: reformulated_query vs cached_questions")
            if reformulated_query and self.question_list:
                approach_1_attempted = True
                question_pairs = []
                for i, cached_question in enumerate(self.question_list):
                    if cached_question and len(cached_question.strip()) > 0:
                        question_pairs.append((reformulated_query, cached_question))
                
                if question_pairs:
                    print(f"[CACHE_SYSTEM] Created {len(question_pairs)} valid question pairs for reranking")
                    
                    if hasattr(reranker, "rerank") and FLASHRANK_AVAILABLE:
                        # FlashRank path - batch rerank (same as Approach 2)
                        print("[CACHE_SYSTEM] Using FlashRank.rerank path for question matching")
                        
                        passages = []
                        for i, cached_question in enumerate(self.question_list):
                            if cached_question and len(cached_question.strip()) > 0:
                                passages.append({"id": f"question_{i}", "text": cached_question})
                        
                        if passages:
                            request = RerankRequest(query=reformulated_query, passages=passages)
                            ranked = reranker.rerank(request)
                            
                            if ranked:
                                # FlashRank returns: {"id": "...", "score": float} or {"id": "...", "relevance_score": float}
                                score_field = None
                                if len(ranked) > 0:
                                    first_result = ranked[0]
                                    if isinstance(first_result, dict):
                                        if "score" in first_result:
                                            score_field = "score"
                                        elif "relevance_score" in first_result:
                                            score_field = "relevance_score"
                                
                                if score_field:
                                    # Create score list aligned with question_list order
                                    score_map = {}
                                    for r in ranked:
                                        r_id = r.get("id")
                                        r_score = r.get(score_field, 0.0)
                                        if r_id:
                                            score_map[r_id] = float(r_score)
                                    
                                    scores_q = []
                                    for i in range(len(self.question_list)):
                                        question_id = f"question_{i}"
                                        score_val = score_map.get(question_id, 0.0)
                                        scores_q.append(score_val)
                                    
                                    if scores_q:
                                        best_match_1_idx = np.argmax(scores_q)
                                        best_match_1_score = float(scores_q[best_match_1_idx])
                                        print(f"[CACHE_SYSTEM] Best question match (FlashRank): idx={best_match_1_idx}, score={best_match_1_score:.3f}")
                    
                    elif hasattr(reranker, "predict"):
                        # CrossEncoder path - batch predict
                        print("[CACHE_SYSTEM] Using CrossEncoder.predict path for question matching")
                        scores_q = reranker.predict(question_pairs)
                        
                        # Handle numpy array or list
                        if hasattr(scores_q, 'tolist'):
                            scores_q = scores_q.tolist()
                        if not isinstance(scores_q, list):
                            scores_q = list(scores_q) if hasattr(scores_q, '__iter__') else [scores_q]
                        
                        if scores_q:
                            best_match_1_idx = np.argmax(scores_q)
                            best_match_1_score = float(scores_q[best_match_1_idx])
                            print(f"[CACHE_SYSTEM] Best question match (CrossEncoder): idx={best_match_1_idx}, score={best_match_1_score:.3f}")
            
            # APPROACH 2: Compare sample_answer with cached answers (same logic as bert_filter_candidates)
            print("\n[CACHE_SYSTEM] APPROACH 2: sample_answer vs cached_answers")
            if sample_answer and self.answer_list:
                approach_2_attempted = True
                # Use sample_answer for comparison
                comparison_text = sample_answer
                
                answer_pairs = []
                for i, cached_answer in enumerate(self.answer_list):
                    if cached_answer and len(cached_answer.strip()) > 0:
                        answer_pairs.append((comparison_text, cached_answer))
                
                if answer_pairs:
                    print(f"[CACHE_SYSTEM] Created {len(answer_pairs)} valid answer pairs for reranking")
                    
                    if hasattr(reranker, "rerank") and FLASHRANK_AVAILABLE:
                        # FlashRank path - batch rerank (same as bert_filter_candidates)
                        print("[CACHE_SYSTEM] Using FlashRank.rerank path for answer matching")
                        
                        # For FlashRank, use reformulated_query if sample_answer is too long
                        flashrank_query = reformulated_query if len(comparison_text) > 200 else comparison_text
                        print(f"[CACHE_SYSTEM] Using {'reformulated_query' if len(comparison_text) > 200 else 'sample_answer'} for FlashRank (len={len(flashrank_query)})")
                        
                        passages = []
                        for i, cached_answer in enumerate(self.answer_list):
                            if cached_answer and len(cached_answer.strip()) > 0:
                                passages.append({"id": f"cache_{i}", "text": cached_answer})
                        
                        if passages:
                            request = RerankRequest(query=flashrank_query, passages=passages)
                            ranked = reranker.rerank(request)
                            
                            if ranked:
                                # FlashRank returns: {"id": "...", "score": float} or {"id": "...", "relevance_score": float}
                                score_field = None
                                if len(ranked) > 0:
                                    first_result = ranked[0]
                                    if isinstance(first_result, dict):
                                        if "score" in first_result:
                                            score_field = "score"
                                        elif "relevance_score" in first_result:
                                            score_field = "relevance_score"
                                
                                if score_field:
                                    # Create score list aligned with answer_list order
                                    score_map = {}
                                    for r in ranked:
                                        r_id = r.get("id")
                                        r_score = r.get(score_field, 0.0)
                                        if r_id:
                                            score_map[r_id] = float(r_score)
                                    
                                    scores_a = []
                                    for i in range(len(self.answer_list)):
                                        cache_id = f"cache_{i}"
                                        score_val = score_map.get(cache_id, 0.0)
                                        scores_a.append(score_val)
                                    
                                    if scores_a:
                                        best_match_2_idx = np.argmax(scores_a)
                                        best_match_2_score = float(scores_a[best_match_2_idx])
                                        print(f"[CACHE_SYSTEM] Best answer match (FlashRank): idx={best_match_2_idx}, score={best_match_2_score:.3f}")
                    
                    elif hasattr(reranker, "predict"):
                        # CrossEncoder path - batch predict (same as bert_filter_candidates)
                        print("[CACHE_SYSTEM] Using CrossEncoder.predict path for answer matching")
                        scores_a = reranker.predict(answer_pairs)
                        
                        # Handle numpy array or list
                        if hasattr(scores_a, 'tolist'):
                            scores_a = scores_a.tolist()
                        if not isinstance(scores_a, list):
                            scores_a = list(scores_a) if hasattr(scores_a, '__iter__') else [scores_a]
                        
                        if scores_a:
                            best_match_2_idx = np.argmax(scores_a)
                            best_match_2_score = float(scores_a[best_match_2_idx])
                            print(f"[CACHE_SYSTEM] Best answer match (CrossEncoder): idx={best_match_2_idx}, score={best_match_2_score:.3f}")
            
            # Check if BOTH approaches meet threshold (both must be >= 0.60)
            cache_end = time.perf_counter()
            print(f"\n[CACHE_SYSTEM] Reranker computation took {cache_end - cache_start:.3f} seconds")
            approach_1_status = f"best_score={best_match_1_score:.3f}, idx={best_match_1_idx}" if approach_1_attempted else "NOT ATTEMPTED (empty reformulated_query or no cached questions)"
            approach_2_status = f"best_score={best_match_2_score:.3f}, idx={best_match_2_idx}" if approach_2_attempted else "NOT ATTEMPTED (empty sample_answer or no cached answers)"
            print(f"[CACHE_SYSTEM] Approach 1 (query-to-question): {approach_1_status}")
            print(f"[CACHE_SYSTEM] Approach 2 (answer-to-answer): {approach_2_status}")
            
            # Both checks must be >= 0.60 to use cache
            # If either approach wasn't attempted or scored below 0.60, proceed to retrieval
            if approach_1_attempted and approach_2_attempted and best_match_1_score >= 0.60 and best_match_2_score >= 0.60:
                # Both checks passed - use cache answer
                # Use the answer-to-answer match (approach 2) as it's more reliable for semantic matching
                best_idx = best_match_2_idx
                best_similarity = best_match_2_score
                
                print(f"\n[CACHE_SYSTEM] BOTH checks passed threshold 0.60:")
                print(f"[CACHE_SYSTEM]   Approach 1 (query-to-question): {best_match_1_score:.3f} >= 0.60 ✓")
                print(f"[CACHE_SYSTEM]   Approach 2 (answer-to-answer): {best_match_2_score:.3f} >= 0.60 ✓")
                print(f"[CACHE_SYSTEM] Using cached answer at idx={best_idx}")
                cached_q_preview = self.question_list[best_idx][:100] if best_idx < len(self.question_list) else "(none)"
                print(f"[CACHE_SYSTEM] Cached question: {cached_q_preview}...")
                print(f"[CACHE_SYSTEM] Cached answer preview: {self.answer_list[best_idx][:100]}...")
                
                # Step 2: Use LLM to verify and reframe the cached answer
                print(f"[CACHE_SYSTEM] Using LLM to verify and reframe cached answer...")
                cached_question = self.question_list[best_idx] if best_idx < len(self.question_list) else ""
                cached_answer = self.answer_list[best_idx]
                
                # Use LLM to verify and reframe the cached answer
                reframed_answer = self._verify_and_reframe_cache(reformulated_query, cached_question, cached_answer)
                
                if reframed_answer:
                    print("[CACHE_SYSTEM] LLM verified and reframed cached answer successfully")
                    print("="*60)
                    return reframed_answer
                else:
                    print("[CACHE_SYSTEM] LLM determined cached answer cannot answer the query")
                    print("="*60)
                    return None
            else:
                # At least one check failed or wasn't attempted - proceed to retrieval
                print(f"\n[CACHE_SYSTEM] At least one check below threshold 0.60 or not attempted - proceeding to retrieval")
                if approach_1_attempted:
                    print(f"[CACHE_SYSTEM] Approach 1 (query-to-question): {best_match_1_score:.3f} {'>= 0.60 ✓' if best_match_1_score >= 0.60 else '< 0.60 ✗'}")
                else:
                    print(f"[CACHE_SYSTEM] Approach 1 (query-to-question): NOT ATTEMPTED ✗")
                if approach_2_attempted:
                    print(f"[CACHE_SYSTEM] Approach 2 (answer-to-answer): {best_match_2_score:.3f} {'>= 0.60 ✓' if best_match_2_score >= 0.60 else '< 0.60 ✗'}")
                else:
                    print(f"[CACHE_SYSTEM] Approach 2 (answer-to-answer): NOT ATTEMPTED ✗")
                print("="*60)
                return None
                
        except Exception as e:
            print(f"[CACHE_SYSTEM] Error during cache check: {e}")
            import traceback
            traceback.print_exc()
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
    chat_history_context_for_llm, last_3_qa_pairs = get_chat_context(history_pairs, current_summary)
    t2 = time.perf_counter()
    timer.mark("get_chat_context")
    print(f"chat context build took {t2 - t1:.2f} secs")
    print(f"[pipeline] chat_context_len={len(chat_history_context_for_llm)}")
    print(f"[pipeline] last_3_qa_pairs={last_3_qa_pairs[:200] if last_3_qa_pairs else '(none)'}")
    
    # Combined classification + reformulation using Gemini (OPTIMIZED: Single LLM call instead of two)
    print("\n" + "="*80)
    print("STEP 1+2: CLASSIFICATION + REFORMULATION (GEMINI OPTIMIZED)")
    print("="*80)
    print(f"[PIPELINE] Starting classify_and_reformulate for: {user_message[:50]}...")
    
    label, reason, rewritten, correction, sample_answer = classify_and_reformulate(chat_history_context_for_llm, user_message, last_3_qa_pairs)
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
        total_time = t_end - start_time
        timer.total("request")
        print(f"total took {total_time:.2f} secs")
        return {"answer": reply, "intent": "chitchat", "follow_up": None, "total_time": round(total_time, 3)}

    # Skip the old reformulation step since it's now combined above

    # ---- CACHE CHECK (NEW: BERT + LLM APPROACH) ----
    print("[pipeline] Step 3: Cache check using reranker (same logic as bert_filter_candidates)...")
    cached_answer = None
    if cache_system:
        cache_start = time.perf_counter()
        # Use sample_answer for comparison (same approach as bert_filter_candidates)
        cached_answer = cache_system.check_cache(sample_answer, rewritten)
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
        total_time = t_end - start_time
        timer.total("request")
        print(f"total took {total_time:.2f} secs")
        
        # Return cached response with video URL
        response = {"answer": cached_answer, "intent": "answer", "follow_up": None, "total_time": round(total_time, 3)}
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

    # Edge case: No candidates retrieved
    if not candidates or len(candidates) == 0:
        print("[pipeline] No candidates retrieved from hybrid_retrieve")
        msg = "Sorry I am not aware this topic, Please ask any other medical related terminologies"
        background_tasks.add_task(_background_update_and_save, user_id, user_message, msg, "no_context", history_pairs, current_summary)
        t_end = time.perf_counter()
        total_time = t_end - start_time
        timer.total("request")
        print(f"total took {total_time:.2f} secs")
        return {"answer": msg, "intent": "no_context", "follow_up": None, "video_url": None, "total_time": round(total_time, 3)}

    print("[pipeline] Step 5: bert_filter_candidates")
    # Use sample_answer for better semantic matching (answer-to-answer comparison)
    filtered = bert_filter_candidates(candidates, sample_answer, rewritten, min_score=0.35)
    t5 = time.perf_counter()
    timer.mark("bert_filter_candidates")
    print(f"BERT filtering took {t5 - t4:.2f} secs")
    print(f"[pipeline] BERT filter -> answers={len(filtered['answer_chunks'])} followups={len(filtered['followup_chunks'])}")

    # ---- BERT FILTER → SYNTHESIZE ----
    if filtered["answer_chunks"]:
        top4 = filtered["answer_chunks"]
        followup_candidates = filtered["followup_chunks"]

        followup_q = ""
        if followup_candidates: # Add this check
            # Combine FULL TEXT of up to 2 followup candidates to give LLM enough context for topic extraction
            followup_texts = []
            for fc in followup_candidates[:2]:
                # Get full text content - use text field (not just section, as section might be too short)
                chunk_text = fc.get("text", "")
                if chunk_text and len(chunk_text.strip()) > 0:
                    # Use full text (up to 500 chars per chunk to avoid too long, but ensure we have enough)
                    truncated_text = chunk_text[:500] if len(chunk_text) > 500 else chunk_text
                    followup_texts.append(truncated_text)
                    print(f"[pipeline] Followup chunk {len(followup_texts)}: {len(truncated_text)} chars, preview: {truncated_text[:100]}...")
                else:
                    print(f"[pipeline] WARNING: Followup candidate has empty text, skipping")
            
            # Combine followup contexts with clear separation
            if followup_texts:
                followup_q = "\n\n".join(followup_texts)
                print(f"[pipeline] Followup context extracted: {len(followup_q)} chars from {len(followup_texts)} chunks")
                print(f"[pipeline] Followup context preview: {followup_q[:200]}...")
            else:
                print("[pipeline] WARNING: No valid followup text extracted from candidates")

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
        total_time = t_end - start_time
        timer.total("request")
        print(f"total took {total_time:.2f} secs")
        
        # Return response with video URL
        response = {"answer": answer, "intent": "answer", "follow_up": followup_q if followup_q else None, "total_time": round(total_time, 3)}
        if video_url:
            response["video_url"] = video_url
        else:
            response["video_url"] = None
        
        return response

    else:
        # Edge case: No qualified chunks found
        print("[pipeline] No qualified chunks found (all candidates below threshold or empty)")
        msg = "Sorry I am not aware this topic, Please ask any other medical related terminologies"
        # Schedule full update+save in background for no_context
        print("[pipeline] schedule background save: no_context")
        background_tasks.add_task(_background_update_and_save, user_id, user_message, msg, "no_context", history_pairs, current_summary)
        print("[pipeline] done with no_context")
        t_end = time.perf_counter()
        total_time = t_end - start_time
        timer.total("request")
        print(f"total took {total_time:.2f} secs")
        
        # Return response with video URL (None for no_context)
        return {"answer": msg, "intent": "no_context", "follow_up": None, "video_url": None, "total_time": round(total_time, 3)}


# -------------------- API ENDPOINTS --------------------
@app.on_event("startup")
async def startup_event():
    global embedding_model, reranker, chromadb_collection, llm, summarizer_llm, reformulate_llm, classifier_llm, gemini_llm, EMBED_DIM, video_system, cache_system
    print("[startup] Initializing models and ChromaDB client...")
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

    # Initialize ChromaDB
    if not CHROMADB_AVAILABLE:
        raise RuntimeError("ChromaDB not available. Install with: pip install chromadb")
    
    VECTORSTORE_DIR = Path("FILES/local_vectorstore")
    if not VECTORSTORE_DIR.exists():
        print(f"[startup] WARNING: ChromaDB not found at {VECTORSTORE_DIR}")
        print("[startup] Please run setup_local_vectorstore.py first to create the vector store")
        raise RuntimeError("ChromaDB vector store not found. Run setup_local_vectorstore.py first.")
    
    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    t.mark("init_chromadb_client")
    
    try:
        chromadb_collection = client.get_collection(name="medical_documents")
        collection_count = chromadb_collection.count()
        if collection_count == 0:
            print(f"[startup] WARNING: ChromaDB collection loaded but is EMPTY ({collection_count} chunks)")
            print("[startup] Please run setup_local_vectorstore.py to populate the collection")
        else:
            print(f"[startup] ChromaDB collection loaded successfully with {collection_count} chunks")
        
        # Verify collection has the expected structure
        try:
            # Try a dummy query to verify collection is functional
            test_embedding = [0.0] * EMBED_DIM
            test_results = chromadb_collection.query(
                query_embeddings=[test_embedding],
                n_results=1
            )
            print(f"[startup] ChromaDB collection verified and ready")
        except Exception as e:
            print(f"[startup] WARNING: ChromaDB collection verification failed: {e}")
            print("[startup] Collection may not be properly initialized")
    except Exception as e:
        print(f"[startup] ERROR: Could not load ChromaDB collection: {e}")
        print("[startup] Please run setup_local_vectorstore.py first to create the vector store")
        raise RuntimeError(f"ChromaDB collection not found: {e}")
    
    t.mark("chromadb_collection_ready")

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