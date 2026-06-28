"""
honeypot_detector.py
---------------------
V2 Architecture: Probabilistic Honeypot Detection.
Instead of hard rejections, returns a honeypot_probability score (0.0 to 1.0).
"""

import re
from datetime import date
from jd_config import TECHNOLOGY_TIMELINES

def _parse_year(val) -> int | None:
    if not val:
        return None
    try:
        return int(str(val)[:4])
    except (ValueError, TypeError):
        return None

def _today_year() -> int:
    return date.today().year

# ── Feature checks ─────────────────────────────────────────────────────────────

def _check_exp_vs_career(candidate: dict) -> float:
    """Gap between stated experience and career duration."""
    stated = candidate["profile"].get("years_of_experience", 0) or 0
    career = candidate.get("career_history", [])
    total_months = sum(ch.get("duration_months", 0) or 0 for ch in career)
    total_years = total_months / 12.0
    
    if stated > total_years + 8 and stated > 8:
        return 0.8
    if stated > total_years + 5 and stated > 5:
        return 0.4
    return 0.0

def _check_edu_before_career(candidate: dict) -> float:
    """Education end_year much later than earliest career start_date."""
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])

    start_years = []
    for ch in career:
        yr = _parse_year(ch.get("start_date"))
        if yr:
            start_years.append(yr)

    if not start_years:
        return 0.0

    earliest_work = min(start_years)
    for edu in education:
        end_yr = edu.get("end_year")
        if end_yr and isinstance(end_yr, int):
            if earliest_work < end_yr - 6:
                return 0.9
            if earliest_work < end_yr - 4:
                return 0.5
    return 0.0

def _check_advanced_zero_duration(candidate: dict) -> float:
    skills = candidate.get("skills", [])
    zero_dur_adv = sum(
        1 for s in skills
        if s.get("proficiency") == "advanced" and (s.get("duration_months") or 0) == 0
    )
    if zero_dur_adv >= 5:
        return 0.7
    if zero_dur_adv >= 3:
        return 0.4
    return 0.0

def _check_future_dates(candidate: dict) -> float:
    today = _today_year()
    for ch in candidate.get("career_history", []):
        yr = _parse_year(ch.get("start_date"))
        if yr and yr > today:
            return 1.0  # Hard penalty
    return 0.0

def _check_timeline_consistency(candidate: dict) -> float:
    """
    Checks if job description explicitly claims usage of tech before it existed.
    """
    prob = 0.0
    for ch in candidate.get("career_history", []):
        yr = _parse_year(ch.get("start_date"))
        if not yr:
            continue
            
        desc = (ch.get("description", "") or "").lower()
        title = (ch.get("title", "") or "").lower()
        text = f"{title} {desc}"
        
        for tech, release_year in TECHNOLOGY_TIMELINES.items():
            if yr < release_year - 1:  # Allow 1 year leeway for betas
                # Require explicit usage claim in this specific job
                # e.g., "built with tensorflow", "used transformers"
                usage_patterns = [
                    f"built .* {tech}",
                    f"used {tech}",
                    f"implemented .* {tech}",
                    f"developed .* {tech}",
                    f"architected .* {tech}",
                    f"deployed {tech}",
                ]
                for p in usage_patterns:
                    if re.search(p, text):
                        prob += 0.4  # Compound probabilities if multiple impossible claims
    return min(prob, 1.0)

def _check_impossible_progression(candidate: dict) -> float:
    """Checks for rapid jumps from junior to extreme senior roles."""
    career = candidate.get("career_history", [])
    if len(career) < 2:
        return 0.0
        
    sorted_career = []
    for ch in career:
        yr = _parse_year(ch.get("start_date"))
        if yr:
            sorted_career.append((yr, (ch.get("title") or "").lower()))
            
    sorted_career.sort(key=lambda x: x[0])
    
    prob = 0.0
    for i in range(len(sorted_career) - 1):
        yr1, title1 = sorted_career[i]
        yr2, title2 = sorted_career[i+1]
        
        is_junior1 = any(w in title1 for w in ["intern", "trainee", "junior"])
        is_senior2 = any(w in title2 for w in ["principal", "architect", "director", "vp", "head"])
        
        if is_junior1 and is_senior2 and (yr2 - yr1) <= 2:
            prob += 0.8
            
    return min(prob, 1.0)

# ── Public API ────────────────────────────────────────────────────────────────

def get_honeypot_probability(candidate: dict) -> float:
    """
    Returns a probability score (0.0 to 1.0) that the candidate is a honeypot.
    """
    scores = [
        _check_exp_vs_career(candidate),
        _check_edu_before_career(candidate),
        _check_advanced_zero_duration(candidate),
        _check_future_dates(candidate),
        _check_timeline_consistency(candidate),
        _check_impossible_progression(candidate)
    ]
    
    # Calculate probability 
    # If any single check is 1.0, it's 1.0. 
    # Otherwise, it compounds loosely.
    max_score = max(scores)
    sum_score = sum(scores)
    
    prob = min(max_score + (sum_score - max_score) * 0.5, 1.0)
    return round(prob, 4)

def honeypot_flags(candidate: dict) -> list[str]:
    flags = []
    if _check_exp_vs_career(candidate) > 0.3:
        flags.append("Experience duration mismatch")
    if _check_edu_before_career(candidate) > 0.3:
        flags.append("Education after work start")
    if _check_advanced_zero_duration(candidate) > 0.3:
        flags.append("Advanced skills with 0 months")
    if _check_future_dates(candidate) > 0.3:
        flags.append("Future employment dates")
    if _check_timeline_consistency(candidate) > 0.3:
        flags.append("Impossible technology timeline")
    if _check_impossible_progression(candidate) > 0.3:
        flags.append("Impossible career progression")
    return flags

def is_honeypot(candidate: dict) -> bool:
    return get_honeypot_probability(candidate) > 0.5
