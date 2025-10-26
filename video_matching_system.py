#!/usr/bin/env python3
"""
Video Matching System for Medical Answers
Optimized approach with pre-filtering + LLM scoring
"""
import pandas as pd
import json
import time
from typing import Dict, List, Tuple, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VideoMatchingSystem:
    def __init__(self, video_file_path: str = "D:\\RHL-WH\\RHL-FASTAPI\\FILES\\video_link_topic.xlsx"):
        """Initialize the video matching system"""
        self.video_file_path = video_file_path
        self.topic_dict = {}  # topic -> index
        self.url_dict = {}    # index -> URL
        self.gemini_llm = None
        
        # Load video data
        self._load_video_data()
        
        # Initialize Gemini LLM
        self._init_gemini()
    
    def _load_video_data(self):
        """Load and preprocess video data"""
        print("="*60)
        print("VIDEO MATCHING SYSTEM - DATA LOADING")
        print("="*60)
        
        try:
            df = pd.read_excel(self.video_file_path)
            print(f"Loaded {len(df)} videos from {self.video_file_path}")
            
            # Create dictionaries
            for idx, row in df.iterrows():
                topic = row['video_topic'].strip()
                url = row['URL'].strip()
                
                if topic and url:
                    self.topic_dict[topic] = idx
                    self.url_dict[idx] = url
            
            print(f"Created topic_dict with {len(self.topic_dict)} topics")
            print(f"Created url_dict with {len(self.url_dict)} URLs")
            
            # Show sample topics
            print("\nSample topics:")
            for i, (topic, idx) in enumerate(list(self.topic_dict.items())[:3]):
                print(f"{i+1}. {topic[:100]}...")
                
        except Exception as e:
            print(f"Error loading video data: {e}")
            self.topic_dict = {}
            self.url_dict = {}
    
    def _init_gemini(self):
        """Initialize Gemini LLM"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("Warning: GOOGLE_API_KEY not found. Video matching will use word-only matching.")
                return
            
            self.gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                api_key=api_key,
                temperature=0.1,
                max_tokens=200,
                timeout=10
            )
            print("Gemini LLM initialized successfully")
            
        except Exception as e:
            print(f"Error initializing Gemini: {e}")
            self.gemini_llm = None
    
    def pre_filter_topics(self, answer: str, min_matches: int = 2) -> List[Tuple[int, int]]:
        """Fast word matching to reduce candidates"""
        print(f"[VIDEO_MATCH] Pre-filtering topics for answer: {answer[:100]}...")
        
        candidates = []
        answer_words = set(answer.lower().split())
        
        for topic, idx in self.topic_dict.items():
            # Split topic by comma and get individual words
            topic_words = set()
            for term in topic.split(','):
                topic_words.update(term.strip().lower().split())
            
            # Count matches
            matches = len(answer_words.intersection(topic_words))
            
            if matches >= min_matches:
                candidates.append((idx, matches))
        
        # Sort by matches (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"[VIDEO_MATCH] Found {len(candidates)} candidates with >= {min_matches} matches")
        return candidates
    
    def llm_score_candidates(self, answer: str, candidates: List[Tuple[int, int]]) -> Optional[int]:
        """Use LLM to score top candidates"""
        if not self.gemini_llm or len(candidates) <= 1:
            return candidates[0][0] if candidates else None
        
        print(f"[VIDEO_MATCH] LLM scoring {len(candidates)} candidates")
        
        # Create prompt with top candidates
        topic_list = []
        for idx, matches in candidates[:10]:  # Limit to top 10 for efficiency
            topic = list(self.topic_dict.keys())[list(self.topic_dict.values()).index(idx)]
            topic_list.append(f"{idx}: {topic}")
        
        prompt = f"""Score these video topics against the medical answer (0-100 each):

Answer: {answer}

Topics:
{chr(10).join(topic_list)}

Return JSON: {{"scores": [85, 92, 45, ...]}}"""
        
        try:
            start_time = time.time()
            response = self.gemini_llm.invoke([HumanMessage(content=prompt)]).content
            llm_time = time.time() - start_time
            
            print(f"[VIDEO_MATCH] LLM call took {llm_time:.3f}s")
            
            # Parse JSON response
            try:
                # Extract JSON from response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    scores_data = json.loads(json_str)
                    scores = scores_data.get('scores', [])
                    
                    if scores and len(scores) == len(candidates[:10]):
                        # Find best score
                        best_score_idx = scores.index(max(scores))
                        best_candidate_idx = candidates[best_score_idx][0]
                        print(f"[VIDEO_MATCH] Best score: {max(scores)} for candidate {best_candidate_idx}")
                        return best_candidate_idx
                        
            except Exception as e:
                print(f"[VIDEO_MATCH] Error parsing LLM response: {e}")
                print(f"[VIDEO_MATCH] Raw response: {response[:200]}...")
                
        except Exception as e:
            print(f"[VIDEO_MATCH] LLM call failed: {e}")
        
        # Fallback to first candidate
        return candidates[0][0] if candidates else None
    
    def find_relevant_video(self, answer: str) -> Optional[str]:
        """Find relevant video URL for the answer"""
        print("="*60)
        print("VIDEO MATCHING - FINDING RELEVANT VIDEO")
        print("="*60)
        
        if not self.topic_dict:
            print("[VIDEO_MATCH] No video data loaded")
            return None
        
        start_time = time.time()
        
        # Step 1: Pre-filtering (fast)
        candidates = self.pre_filter_topics(answer, min_matches=3)
        
        if not candidates:
            print("[VIDEO_MATCH] No candidates found with >= 3 word matches")
            return None
        
        # Step 2: LLM scoring (if multiple candidates)
        if len(candidates) == 1:
            best_idx = candidates[0][0]
            print(f"[VIDEO_MATCH] Single candidate found: {best_idx}")
        else:
            best_idx = self.llm_score_candidates(answer, candidates)
        
        # Step 3: Get URL
        if best_idx is not None and best_idx in self.url_dict:
            url = self.url_dict[best_idx]
            total_time = time.time() - start_time
            print(f"[VIDEO_MATCH] Found relevant video: {url}")
            print(f"[VIDEO_MATCH] Total time: {total_time:.3f}s")
            return url
        else:
            print("[VIDEO_MATCH] No valid video found")
            return None

# Test the system
def test_video_matching():
    """Test the video matching system"""
    print("="*80)
    print("VIDEO MATCHING SYSTEM TEST")
    print("="*80)
    
    # Initialize system
    video_system = VideoMatchingSystem()
    
    # Test cases
    test_answers = [
        "The patient presented with uterine infection, maternal fever, and foul-smelling discharge. Risk factors include prolonged rupture of membranes and chorioamnionitis.",
        "How to measure baby's temperature using a thermometer?",
        "The newborn needs vitamin K injection to prevent bleeding disorders.",
        "Cord care is essential to prevent umbilical cord infection and omphalitis.",
        "This is about cooking recipes and has nothing to do with medical topics."
    ]
    
    for i, answer in enumerate(test_answers, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {answer[:100]}...")
        print(f"{'='*60}")
        
        url = video_system.find_relevant_video(answer)
        
        if url:
            print(f"FOUND VIDEO: {url}")
        else:
            print("NO VIDEO FOUND")

if __name__ == "__main__":
    test_video_matching()
