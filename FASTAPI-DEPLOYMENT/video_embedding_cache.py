"""
Video Embedding Cache Module
Pre-computes and caches video description embeddings to avoid re-encoding on every request.
"""
import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import pandas as pd

class VideoEmbeddingCache:
    """
    Manages pre-computed embeddings for video descriptions.
    Caches embeddings to disk to avoid re-encoding on every request.
    """
    
    def __init__(self, 
                 video_file_path: str = "./FILES/video_link_topic.xlsx",
                 cache_dir: str = "./FILES/video_cache",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the video embedding cache.
        
        Args:
            video_file_path: Path to Excel file with video data
            cache_dir: Directory to store cached embeddings
            model_name: BERT model name for encoding
        """
        self.video_file_path = video_file_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        
        # Cache file paths
        self.embeddings_cache_file = self.cache_dir / "video_embeddings.pkl"
        self.metadata_cache_file = self.cache_dir / "video_metadata.pkl"
        
        # Loaded data
        self.topic_list: List[str] = []
        self.url_list: List[str] = []
        self.topic_embeddings: Optional[np.ndarray] = None
        self.similarity_model: Optional[SentenceTransformer] = None
        
    def _load_video_data(self) -> Tuple[List[str], List[str]]:
        """Load video data from Excel file."""
        try:
            df = pd.read_excel(self.video_file_path)
            print(f"[VIDEO_CACHE] Loaded {len(df)} videos from {self.video_file_path}")
            
            topic_list = []
            url_list = []
            
            for idx, row in df.iterrows():
                description = row['Description'].strip()
                url = row['URL'].strip()
                
                if description and url:
                    topic_list.append(description)
                    url_list.append(url)
            
            print(f"[VIDEO_CACHE] Created {len(topic_list)} video entries")
            return topic_list, url_list
            
        except Exception as e:
            print(f"[VIDEO_CACHE] Error loading video data: {e}")
            return [], []
    
    def _compute_embeddings(self, topic_list: List[str]) -> np.ndarray:
        """Compute embeddings for all video descriptions."""
        print(f"[VIDEO_CACHE] Computing embeddings for {len(topic_list)} videos...")
        print(f"[VIDEO_CACHE] Loading model: {self.model_name}")
        
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(topic_list, show_progress_bar=True, batch_size=32)
        
        print(f"[VIDEO_CACHE] Computed embeddings shape: {embeddings.shape}")
        return embeddings
    
    def _save_cache(self, topic_list: List[str], url_list: List[str], embeddings: np.ndarray):
        """Save embeddings and metadata to disk."""
        print(f"[VIDEO_CACHE] Saving cache to {self.cache_dir}...")
        
        # Save embeddings
        with open(self.embeddings_cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        # Save metadata (topic_list and url_list)
        metadata = {
            'topic_list': topic_list,
            'url_list': url_list,
            'model_name': self.model_name,
            'num_videos': len(topic_list)
        }
        with open(self.metadata_cache_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"[VIDEO_CACHE] Cache saved successfully")
    
    def _load_cache(self) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[np.ndarray]]:
        """Load cached embeddings and metadata from disk."""
        if not self.embeddings_cache_file.exists() or not self.metadata_cache_file.exists():
            print(f"[VIDEO_CACHE] Cache files not found")
            return None, None, None
        
        try:
            print(f"[VIDEO_CACHE] Loading cache from {self.cache_dir}...")
            
            # Load metadata
            with open(self.metadata_cache_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # Verify model matches
            if metadata.get('model_name') != self.model_name:
                print(f"[VIDEO_CACHE] Model mismatch: cache={metadata.get('model_name')}, current={self.model_name}")
                print(f"[VIDEO_CACHE] Will recompute embeddings...")
                return None, None, None
            
            # Load embeddings
            with open(self.embeddings_cache_file, 'rb') as f:
                embeddings = np.array(pickle.load(f))
            
            print(f"[VIDEO_CACHE] Loaded {metadata.get('num_videos', 0)} video embeddings from cache")
            return metadata['topic_list'], metadata['url_list'], embeddings
            
        except Exception as e:
            print(f"[VIDEO_CACHE] Error loading cache: {e}")
            return None, None, None
    
    def initialize(self, force_recompute: bool = False) -> bool:
        """
        Initialize the cache: load from disk or compute new embeddings.
        Also loads the similarity model for answer encoding.
        
        Args:
            force_recompute: If True, recompute embeddings even if cache exists
            
        Returns:
            True if successful, False otherwise
        """
        # Try to load from cache first (unless force_recompute)
        if not force_recompute:
            topic_list, url_list, embeddings = self._load_cache()
            if topic_list is not None and url_list is not None and embeddings is not None:
                self.topic_list = topic_list
                self.url_list = url_list
                self.topic_embeddings = embeddings
                print(f"[VIDEO_CACHE] Successfully loaded from cache")
                # Load model for answer encoding (needed at runtime)
                self._load_model()
                return True
        
        # Cache miss or force_recompute: load data and compute embeddings
        print(f"[VIDEO_CACHE] Computing new embeddings...")
        topic_list, url_list = self._load_video_data()
        
        if not topic_list:
            print(f"[VIDEO_CACHE] No video data found")
            return False
        
        # Compute embeddings
        embeddings = self._compute_embeddings(topic_list)
        
        # Save to cache
        self._save_cache(topic_list, url_list, embeddings)
        
        # Store in memory
        self.topic_list = topic_list
        self.url_list = url_list
        self.topic_embeddings = embeddings
        
        # Load model for answer encoding (needed at runtime)
        self._load_model()
        
        print(f"[VIDEO_CACHE] Initialization complete")
        return True
    
    def _load_model(self):
        """Load the similarity model for answer encoding (called once during initialization)"""
        if self.similarity_model is None:
            print(f"[VIDEO_CACHE] Loading model for answer encoding: {self.model_name}")
            self.similarity_model = SentenceTransformer(self.model_name)
            print(f"[VIDEO_CACHE] Model loaded successfully")
    
    def get_cached_embeddings(self) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Get cached embeddings and metadata.
        
        Returns:
            Tuple of (topic_list, url_list, embeddings)
        """
        if self.topic_embeddings is None:
            raise RuntimeError("Cache not initialized. Call initialize() first.")
        
        return self.topic_list, self.url_list, self.topic_embeddings
    
    def encode_answer(self, answer: str) -> np.ndarray:
        """
        Encode a single answer text (only this needs to be computed at runtime).
        
        Args:
            answer: Answer text to encode
            
        Returns:
            Embedding vector
        """
        # Model should be loaded during initialize(), but fallback if not
        if self.similarity_model is None:
            print(f"[VIDEO_CACHE] WARNING: Model not loaded, loading now (should have been loaded at startup)")
            self._load_model()
        
        return self.similarity_model.encode([answer])[0]
    
    def is_initialized(self) -> bool:
        """Check if cache is initialized."""
        return self.topic_embeddings is not None and len(self.topic_list) > 0


