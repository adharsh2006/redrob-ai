"""
reasoning_generator.py
-----------------------
V2 Architecture: Strictly factual reasoning. No narrative fluff.
"""

def generate_reasoning(
    candidate: dict, 
    fast_features: dict, 
    v2_scores: dict, 
    semantic_match: float
) -> str:
    profile = candidate["profile"]
    exp = profile.get("years_of_experience", 0) or 0
    title = profile.get("current_title", "Engineer")
    
    # Format scores
    ai_depth = v2_scores.get("ai_depth", 0)
    semantic_pct = min(semantic_match * 100, 100)
    seniority = v2_scores.get("seniority", 0)
    
    # Get response rate
    sig = candidate.get("redrob_signals", {}) or {}
    rr = float(sig.get("recruiter_response_rate", 0) or 0) * 100
    
    np_raw = sig.get("notice_period_days")
    try:
        np_days = int(np_raw) if np_raw is not None else "?"
    except (ValueError, TypeError):
        np_days = "?"

    
    # Find top AI skills
    skills = [s.get("name", "") for s in candidate.get("skills", [])]
    ai_skills = [s for s in skills if s.lower() in ["rag", "llm", "search", "recommendation", "ranking", "vector"]]
    skill_str = ", ".join(ai_skills[:3]) if ai_skills else "General AI"
    
    # Factual structure:
    # 8.0 yrs exp (Senior ML Engineer). Semantic Match: 85%. AI Depth Score: 40. 
    # Seniority Score: 35. Signals: 90% response rate, 30d notice. Skills: RAG, LLM.
    
    reasoning = (
        f"{exp:.1f}y exp ({title}). "
        f"Semantic Match: {semantic_pct:.0f}%. "
        f"AI Depth: {ai_depth:.0f}. "
        f"Seniority: {seniority:.0f}. "
        f"Signals: {rr:.0f}% response rate, {np_days}d notice. "
        f"Keywords: {skill_str}."
    )
    
    # If there's a honeypot probability > 0.1, add it factually
    hp_prob = fast_features.get("honeypot_prob", 0)
    if hp_prob > 0.1:
        reasoning += f" [Honeypot Prob: {hp_prob*100:.0f}%]"
        
    return reasoning
