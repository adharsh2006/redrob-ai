"""
feature_builder.py
------------------
V2 Architecture: Feature construction and composite scoring.
"""

from __future__ import annotations
import math
from datetime import date
from jd_config import (
    WEIGHTS, SALARY_MIN_FIT, SALARY_MAX_FIT, PREFERRED_CITIES
)
from honeypot_detector import get_honeypot_probability

def _norm(text) -> str:
    return (text or "").lower().strip()

def _days_since(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        d = date.fromisoformat(str(date_str)[:10])
        return (date.today() - d).days
    except (ValueError, TypeError):
        return None

# ── Stage 1: Fast Features (100K -> 5000) ────────────────────────────────────

def build_fast_features(candidate: dict) -> dict:
    """
    Computes cheap features to filter 100K down to 5000.
    """
    f = {}
    f["candidate_id"] = candidate["candidate_id"]
    
    # 1. Honeypot Prob
    f["honeypot_prob"] = get_honeypot_probability(candidate)
    
    # 2. Behavioral Signals
    sig = candidate.get("redrob_signals", {}) or {}
    rr = float(sig.get("recruiter_response_rate", 0) or 0)
    otw = 1.0 if bool(sig.get("open_to_work_flag", False)) else 0.2
    try:
        np_raw = sig.get("notice_period_days")
        np_days = int(np_raw) if np_raw is not None else 180
    except (ValueError, TypeError):
        np_days = 180
    np_score = 1.0 if np_days <= 30 else (0.8 if np_days <= 60 else (0.4 if np_days <= 90 else 0.1))
    
    # 3. Availability Score
    f["availability"] = 0.6 * np_score + 0.4 * otw
    
    # 4. Behavioral Score
    icr = float(sig.get("interview_completion_rate", 0.5) or 0.5)
    f["behavioral"] = 0.5 * rr + 0.5 * icr
    
    # 5. Fast Keyword Hit (IR / AI basic check)
    text = _norm(candidate["profile"].get("summary", ""))
    for s in candidate.get("skills", []):
        text += " " + _norm(s.get("name", ""))
        
    ai_keywords = {"rag", "llm", "search", "ranking", "recommendation", "machine learning", "nlp", "learning to rank", "vector"}
    hits = sum(1 for kw in ai_keywords if kw in text)
    f["fast_ai_hits"] = min(hits / 3.0, 1.0)
    
    # Stage 1 Heuristic Score (for keeping top 5000)
    # We heavily penalize honeypots, and want good AI hits + decent behavior.
    f["stage1_score"] = (
        (1.0 - f["honeypot_prob"]) * 5.0 + 
        f["fast_ai_hits"] * 3.0 + 
        f["behavioral"] * 1.0
    )
    
    return f

# ── Stage 3: Final Composite Score ───────────────────────────────────────────

def compute_final_score(
    fast_features: dict, 
    v2_career_scores: dict, 
    semantic_match: float
) -> float:
    """
    Calculates the final weighted score for the Top 100 selection.
    FinalScore = (
        0.35 * MiniLMSimilarity
      + 0.25 * AIDepth
      + 0.15 * Seniority
      + 0.10 * ExperienceMaturity
      + 0.10 * Behavioral
      + 0.05 * Availability
    )
    """
    # If it's highly likely a honeypot, kill the score
    if fast_features["honeypot_prob"] > 0.8:
        return 0.0
        
    # Normalize inputs to 0.0 - 1.0 roughly
    ai_depth_norm = min(v2_career_scores["ai_depth"] / 50.0, 1.0)
    seniority_norm = min(v2_career_scores["seniority"] / 50.0, 1.0)
    maturity_norm = v2_career_scores["exp_maturity"] / 100.0  # already 0-100 logistic
    traj_norm = v2_career_scores["career_trajectory"] / 100.0
    
    # We mix trajectory into Seniority & Maturity gently
    combined_seniority = 0.7 * seniority_norm + 0.3 * traj_norm
    
    # Apply penalty for honeypot prob (soft degradation)
    hp_penalty = max(0.0, 1.0 - fast_features["honeypot_prob"])
    
    score = (
        WEIGHTS["semantic_match"] * semantic_match +
        WEIGHTS["ai_depth"] * ai_depth_norm +
        WEIGHTS["seniority"] * combined_seniority +
        WEIGHTS["experience_maturity"] * maturity_norm +
        WEIGHTS["behavioral"] * fast_features["behavioral"] +
        WEIGHTS["availability"] * fast_features["availability"]
    )
    
    final_score = score * hp_penalty
    return round(final_score, 6)
