"""
semantic_matcher.py
-------------------
V2 Architecture: 2-Level Semantic Matching.
- Stage 2: Fast TF-IDF overlap (keeps top 500)
- Stage 3: Deep MiniLM embeddings (ranks top 100)
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Lazy loaded
SentenceTransformer = None

# ── Job Description Target Text ───────────────────────────────────────────────
JD_TEXT = """
Senior AI Engineer Founding Team Recommendation Systems Ranking Models
Information Retrieval Vector Search Semantic Search Learning to Rank
RAG Fine-tuning MLOps Offline Evaluation NDCG AB Testing
Built deployed architected scaled production Python PyTorch
"""

def _get_candidate_text(candidate: dict) -> str:
    """Concatenate skills and career descriptions for semantic matching."""
    text_parts = []
    
    # Add skills
    for s in candidate.get("skills", []):
        text_parts.append(s.get("name", ""))
        
    # Add career
    for ch in candidate.get("career_history", []):
        text_parts.append(ch.get("title", ""))
        text_parts.append(ch.get("description", "") or "")
        
    return " ".join(text_parts).lower()

# ── Stage 2: Fast TF-IDF Matcher ──────────────────────────────────────────────

class FastSemanticMatcher:
    def __init__(self):
        # We use a basic vocabulary focusing on unigrams and bigrams
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self.is_fitted = False
        
    def fit_and_score(self, candidates: list[dict]) -> list[float]:
        """
        Fit TF-IDF on the current batch of candidates + JD.
        Return cosine similarity scores for each candidate vs JD.
        """
        texts = [JD_TEXT.lower()] + [_get_candidate_text(c) for c in candidates]
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        
        jd_vector = tfidf_matrix[0]
        candidate_vectors = tfidf_matrix[1:]
        
        # Calculate cosine similarity
        sims = cosine_similarity(jd_vector, candidate_vectors).flatten()
        return sims.tolist()

# ── Stage 3: Deep MiniLM Matcher ──────────────────────────────────────────────

class DeepSemanticMatcher:
    def __init__(self):
        global SentenceTransformer
        if SentenceTransformer is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("Please pip install sentence-transformers")
                
        # all-MiniLM-L6-v2 is extremely fast on CPU
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.jd_embedding = self.model.encode([JD_TEXT])[0]
        
    def score_candidates(self, candidates: list[dict]) -> list[float]:
        """
        Returns deep semantic cosine similarity against the JD.
        Takes ~0.01s per candidate. (Safe for Top 500 -> 5 seconds).
        """
        if not candidates:
            return []
            
        texts = [_get_candidate_text(c) for c in candidates]
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        
        # Cosine similarity
        # embeddings shape: (N, 384), jd_embedding shape: (384,)
        norm_jd = np.linalg.norm(self.jd_embedding)
        norm_cands = np.linalg.norm(embeddings, axis=1)
        
        sims = np.dot(embeddings, self.jd_embedding) / (norm_cands * norm_jd + 1e-10)
        return sims.tolist()
