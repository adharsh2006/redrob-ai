"""
career_nlp.py
-------------
V2 Architecture: AI Depth Scoring, Seniority, Career Trajectory, and Experience Maturity.
"""

import re
import math
from jd_config import (
    AI_NOUNS, AI_LEVELS, AI_TUTORIAL_KEYWORDS, AI_BUZZWORDS,
    SENIORITY_TITLES, SENIORITY_VERBS
)

def _parse_year(date_str) -> int | None:
    if not date_str:
        return None
    try:
        return int(str(date_str)[:4])
    except (ValueError, TypeError):
        return None

def _norm(text: str) -> str:
    return (text or "").lower()

# ── AI Depth Score ─────────────────────────────────────────────────────────────

def get_ai_depth_score(candidate: dict) -> float:
    """
    4-Level Proximity scoring.
    Looks for verbs near AI nouns in the career history.
    """
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    
    score = 0.0
    highest_level = 0
    
    for ch in career:
        text = _norm(ch.get("title", "")) + " " + _norm(ch.get("description", ""))
        
        # Check for tutorials/toy projects
        if any(w in text for w in AI_TUTORIAL_KEYWORDS):
            score -= 5.0
            continue
            
        # Check proximity of verbs to AI nouns
        # We'll use a simple window or just presence in the same job description.
        # Since job descriptions are short, co-occurrence is a strong signal.
        has_ai_noun = any(n in text for n in AI_NOUNS)
        
        if has_ai_noun:
            job_level = 0
            # Check levels from highest to lowest
            if any(v in text for v in AI_LEVELS["L4_LEADERSHIP"]["verbs"]):
                score += AI_LEVELS["L4_LEADERSHIP"]["score"]
                job_level = 4
            elif any(v in text for v in AI_LEVELS["L3_PRODUCTION"]["verbs"]):
                score += AI_LEVELS["L3_PRODUCTION"]["score"]
                job_level = 3
            elif any(v in text for v in AI_LEVELS["L2_IMPLEMENTATION"]["verbs"]):
                score += AI_LEVELS["L2_IMPLEMENTATION"]["score"]
                job_level = 2
            elif any(v in text for v in AI_LEVELS["L1_AWARENESS"]["verbs"]):
                score += AI_LEVELS["L1_AWARENESS"]["score"]
                job_level = 1
                
            highest_level = max(highest_level, job_level)

    # Buzzword stuffing penalty
    # High AI buzzwords in skills, but no implementation verbs in career history
    skill_names = [_norm(s.get("name", "")) for s in skills]
    buzzword_count = sum(1 for s in skill_names if any(b in s for b in AI_BUZZWORDS))
    
    if buzzword_count >= 2 and highest_level < 2:
        score -= 10.0
        
    # Cap score at 100 for normalization later
    return max(0.0, min(score, 100.0))

# ── Seniority Score ────────────────────────────────────────────────────────────

def get_seniority_score(candidate: dict) -> float:
    """
    Extracts seniority signals from titles and descriptions.
    """
    career = candidate.get("career_history", [])
    score = 0.0
    
    for ch in career:
        title = _norm(ch.get("title", ""))
        desc = _norm(ch.get("description", ""))
        
        # Title hits
        if any(t in title for t in SENIORITY_TITLES):
            score += 15.0
            
        # Description verb hits
        if any(v in desc for v in SENIORITY_VERBS):
            score += 5.0
            
    return min(score, 50.0)

# ── Experience Maturity Score ──────────────────────────────────────────────────

def get_experience_maturity_score(candidate: dict) -> float:
    """
    Calculates cumulative months of validated AI/ML experience.
    Uses a logistic curve to cap the score smoothly (e.g., 60 months = high score).
    """
    career = candidate.get("career_history", [])
    ai_months = 0
    
    for ch in career:
        text = _norm(ch.get("title", "")) + " " + _norm(ch.get("description", ""))
        months = ch.get("duration_months", 0) or 0
        
        # If the job mentions AI/ML concepts, count the months
        if any(n in text for n in AI_NOUNS) or "machine learning" in text or "ml" in text:
            ai_months += months
            
    # Logistic curve: midpoint around 48 months (4 years)
    if ai_months == 0:
        return 0.0
        
    k = 0.1 # steepness
    x0 = 48 # midpoint
    maturity = 100.0 / (1.0 + math.exp(-k * (ai_months - x0)))
    return round(maturity, 2)

# ── Career Trajectory Score ────────────────────────────────────────────────────

def get_career_trajectory_score(candidate: dict) -> float:
    """
    Evaluates title progression.
    Rewards growth. Penalizes flat trajectories or impossible jumps.
    """
    career = candidate.get("career_history", [])
    if len(career) < 2:
        return 50.0  # Neutral if only 1 job
        
    # Sort by start year
    sorted_career = []
    for ch in career:
        yr = _parse_year(ch.get("start_date"))
        if yr:
            sorted_career.append((yr, _norm(ch.get("title", ""))))
            
    sorted_career.sort(key=lambda x: x[0])
    
    score = 50.0 # Base neutral score
    
    for i in range(len(sorted_career) - 1):
        _, title1 = sorted_career[i]
        _, title2 = sorted_career[i+1]
        
        is_jr1 = any(w in title1 for w in ["intern", "junior", "trainee", "associate"])
        is_mid1 = not is_jr1 and not any(w in title1 for w in SENIORITY_TITLES)
        is_sr1 = any(w in title1 for w in SENIORITY_TITLES)
        
        is_jr2 = any(w in title2 for w in ["intern", "junior", "trainee", "associate"])
        is_mid2 = not is_jr2 and not any(w in title2 for w in SENIORITY_TITLES)
        is_sr2 = any(w in title2 for w in SENIORITY_TITLES)
        
        # Progression logic
        if is_jr1 and is_mid2:
            score += 15.0  # Good growth
        elif is_mid1 and is_sr2:
            score += 20.0  # Strong progression to senior
        elif is_jr1 and is_sr2:
            score -= 20.0  # Impossible jump (handled by honeypot, but penalized here too)
        elif is_sr1 and is_sr2:
            score += 10.0  # Sustained leadership
        elif title1 == title2:
            score -= 5.0   # Flat title
            
    return max(0.0, min(score, 100.0))

# ── Combined V2 Processor ─────────────────────────────────────────────────────

def get_v2_career_scores(candidate: dict) -> dict:
    """
    Returns the 4 core scores for Stage 2 processing.
    """
    return {
        "ai_depth": get_ai_depth_score(candidate),
        "seniority": get_seniority_score(candidate),
        "exp_maturity": get_experience_maturity_score(candidate),
        "career_trajectory": get_career_trajectory_score(candidate)
    }
