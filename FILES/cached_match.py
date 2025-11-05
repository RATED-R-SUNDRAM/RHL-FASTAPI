from sentence_transformers import SentenceTransformer
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables and initialize models
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = SentenceTransformer('all-MiniLM-L6-v2')
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=GOOGLE_API_KEY)

def check_cached_answer(reformulated_prompt: str, cache_file_path: str, threshold: float = 0.85) -> str:
    """
    Check if a reformulated prompt matches a cached question and verify with Gemini.
    
    Args:
        reformulated_prompt (str): The reformulated user query.
        cache_file_path (str): Path to Excel file with 'question' and 'answer' columns.
        threshold (float): Minimum similarity score to consider a match (default: 0.85).
    
    Returns:
        str or None: Cached answer if verified, None otherwise.
    """
    # Load cache from Excel
    try:
        cache_df = pd.read_excel(cache_file_path)
        if 'question' not in cache_df.columns or 'answer' not in cache_df.columns:
            return None
    except Exception:
        return None

    # Generate embedding for reformulated prompt
    prompt_emb = model.encode(reformulated_prompt, convert_to_numpy=True)

    # Calculate similarity scores for all cached questions
    best_score = -1
    best_answer = None
    cached_questions = cache_df['question'].tolist()
    cached_answers = cache_df['answer'].tolist()

    for idx, cached_question in enumerate(cached_questions):
        cached_emb = model.encode(cached_question, convert_to_numpy=True)
        similarity = np.dot(prompt_emb, cached_emb) / (np.linalg.norm(prompt_emb) * np.linalg.norm(cached_emb))
        
        if similarity > best_score:
            best_score = similarity
            best_answer = cached_answers[idx]

    # Check if best score exceeds threshold
    if best_score < threshold:
        return None

    # Verify with Gemini LLM
    verification_prompt = f"""
    You are a strict medical assistant verifier. Compare the following two questions and determine if they have the SAME exact medical meaning and intent and answer to one can be used to answer other. Return 'YES' only if they are semantically identical in a medical context, 'NO' otherwise.

    Reformulated Question: {reformulated_prompt}
    Cached Question: {cached_questions[cached_questions.index(cached_df.loc[cache_df['answer'] == best_answer, 'question'].iloc[0])]}

    Response (YES/NO):
    """
    try:
        response = gemini_llm.invoke([{"role": "user", "content": verification_prompt}]).content.strip()
        if response == "YES":
            return best_answer
        return None
    except Exception:
        return None

# Example usage
if __name__ == "__main__":
    prompt = "What are the signs of jaundice?"
    cache_path = "cache.xlsx"
    result = check_cached_answer(prompt, cache_path)
    print(f"Result: {result}")