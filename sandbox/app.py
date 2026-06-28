"""
sandbox/app.py
--------------
Streamlit demo app for the JDMSCR ranker.
Required by the submission spec (Section 10.5):
  'A working hosted environment where the ranker can be run on a small
   candidate sample.'

Deploy to: Streamlit Cloud / HuggingFace Spaces
"""

import json
import sys
import csv
import io
import os
from pathlib import Path
import streamlit as st

# Add src dir to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from honeypot_detector import is_honeypot, honeypot_flags
from feature_builder import build_fast_features, compute_final_score
from career_nlp import get_v2_career_scores
from reasoning_generator import generate_reasoning
from semantic_matcher import FastSemanticMatcher, DeepSemanticMatcher

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="JDMSCR — Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)

# ── Styling ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stApp { background-color: #0f0f1a; color: #e2e8f0; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e3a 0%, #252545 100%);
        border: 1px solid #3a3a6a;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    .rank-badge {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border-radius: 50%;
        width: 36px; height: 36px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 14px;
    }
    .honeypot-badge {
        background: #ef4444;
        color: white;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: bold;
    }
    .score-bar {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        height: 6px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
# 🎯 JDMSCR — JD-Aware Multi-Signal Contextual Ranker
### Redrob INDIA RUNS — Data & AI Challenge Demo

> Upload a **sample candidates JSON file** (e.g., `sample_candidates.json`) to see the ranker in action.  
> The ranker uses **career history NLP + temporal AI credibility + behavioral signals** — not just skill keywords.
""")

st.divider()

# ── File Upload ────────────────────────────────────────────────────────────────

col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader(
        "Upload candidates (JSON array or JSONL)",
        type=["json", "jsonl"],
        help="Upload sample_candidates.json or any subset of candidates.jsonl",
    )

with col2:
    top_n = st.slider("Top N to rank", min_value=5, max_value=50, value=10)

# ── Run ranker ────────────────────────────────────────────────────────────────

if uploaded:
    raw = uploaded.read().decode("utf-8")

    # Parse JSON or JSONL
    candidates = []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            candidates = data
        else:
            candidates = [data]
    except json.JSONDecodeError:
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not candidates:
        st.error("Could not parse any candidates from the uploaded file.")
        st.stop()

    st.success(f"✅ Loaded **{len(candidates)}** candidates")

    # Score all candidates using V2 pipeline
    scored = []
    
    fast_feats = [build_fast_features(c) for c in candidates]
    v2_feats = [get_v2_career_scores(c) for c in candidates]
    
    try:
        deep_matcher = DeepSemanticMatcher()
        semantic_scores = deep_matcher.score_candidates(candidates)
    except Exception as e:
        fast_matcher = FastSemanticMatcher()
        semantic_scores = fast_matcher.fit_and_score(candidates)
        
    for idx, c in enumerate(candidates):
        fast_feat = fast_feats[idx]
        v2_feat = v2_feats[idx]
        sem_score = semantic_scores[idx]
        
        final_score = compute_final_score(fast_feat, v2_feat, sem_score)
        
        # Build UI features dict
        ui_features = {
            "score": final_score,
            "career_evidence": v2_feat["ai_depth"] / 10.0,
            "temporal_cred": v2_feat["exp_maturity"] / 10.0,
            "hirability": fast_feat["behavioral"],
            "location_score": 1.0 if c["profile"].get("country") == "India" else 0.2,
            "is_honeypot": fast_feat["honeypot_prob"] > 0.5,
            "india_based": c["profile"].get("country") == "India",
            "fast_feat": fast_feat,
            "v2_feat": v2_feat,
            "sem_score": sem_score
        }
        scored.append((final_score, c["candidate_id"], ui_features, c))

    # Sort: descending score, then candidate_id ascending as tie-break
    scored.sort(key=lambda x: (-x[0], x[1]))

    top = scored[:top_n]

    # ── Metrics row ───────────────────────────────────────────────────────────

    total = len(candidates)
    hp_count = sum(1 for _, _, f, _ in scored if f.get("is_honeypot"))
    india_count = sum(1 for _, _, f, _ in scored if f.get("india_based"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Candidates", total)
    m2.metric("Honeypots Detected", hp_count, delta=f"{100*hp_count/total:.1f}% of pool")
    m3.metric("India-Based", india_count, delta=f"{100*india_count/total:.1f}%")
    m4.metric("Top Score", f"{top[0][0]:.4f}" if top else "—")

    st.divider()

    # ── Top N table ───────────────────────────────────────────────────────────

    st.subheader(f"🏆 Top {top_n} Ranked Candidates")

    for rank_idx, (score, cid, features, candidate) in enumerate(top, start=1):
        profile = candidate["profile"]
        sig = candidate.get("redrob_signals", {}) or {}
        hp_flags = honeypot_flags(candidate)

        reasoning = generate_reasoning(
            candidate, 
            features["fast_feat"], 
            features["v2_feat"], 
            features["sem_score"]
        )

        with st.expander(
            f"#{rank_idx}  {cid}  —  {profile.get('current_title', 'Unknown')}  "
            f"|  Score: {score:.4f}  "
            f"|  {'🇮🇳' if profile.get('country') == 'India' else '🌍'}  "
            f"{'🚨 HONEYPOT' if hp_flags else ''}",
            expanded=(rank_idx <= 3),
        ):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Profile**")
                st.write(f"🏷️ {profile.get('current_title', 'N/A')}")
                st.write(f"📍 {profile.get('location', 'N/A')}, {profile.get('country', '')}")
                st.write(f"⏱️ {profile.get('years_of_experience', 0):.1f} years exp")
                st.write(f"🏢 {profile.get('current_company', 'N/A')}")

            with c2:
                st.markdown("**Key Scores**")
                st.progress(features.get("career_evidence", 0) / 10, text=f"Career Evidence: {features.get('career_evidence', 0):.2f}/10")
                st.progress(features.get("temporal_cred", 0) / 10, text=f"Temporal Credibility: {features.get('temporal_cred', 0):.2f}/10")
                st.progress(features.get("hirability", 0), text=f"Hirability: {features.get('hirability', 0):.2f}")
                st.progress(features.get("location_score", 0), text=f"Location Fit: {features.get('location_score', 0):.2f}")

            with c3:
                st.markdown("**Behavioral Signals**")
                st.write(f"📬 Response Rate: {sig.get('recruiter_response_rate', 0):.0%}")
                st.write(f"📅 Notice Period: {sig.get('notice_period_days', '?')}d")
                st.write(f"🔓 Open to Work: {'Yes' if sig.get('open_to_work_flag') else 'No'}")
                gh = sig.get("github_activity_score", -1)
                st.write(f"💻 GitHub Score: {'N/A' if gh == -1 else gh}")

            if hp_flags:
                st.error(f"🚨 Honeypot Flags: {', '.join(hp_flags)}")

            st.markdown(f"**💬 Reasoning:** _{reasoning}_")

    # ── Download CSV ──────────────────────────────────────────────────────────

    st.divider()
    st.subheader("📥 Download Submission CSV")

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])

    prev_score = None
    for rank_idx, (score, cid, features, candidate) in enumerate(top, start=1):
        if prev_score is not None and score > prev_score:
            score = prev_score
        prev_score = score
        reasoning = generate_reasoning(
            candidate, 
            features["fast_feat"], 
            features["v2_feat"], 
            features["sem_score"]
        )
        writer.writerow([cid, rank_idx, round(score, 4), reasoning])

    st.download_button(
        "Download CSV",
        data=csv_buffer.getvalue(),
        file_name="sample_submission.csv",
        mime="text/csv",
    )

else:
    # Demo mode with instructions
    st.info("👆 Upload `sample_candidates.json` from the challenge bundle to see the ranker in action.")

    st.markdown("""
    ### How JDMSCR Works

    | Feature Layer | What It Does | Why It Matters |
    |---|---|---|
    | 🔍 **Career History NLP** | Reads job descriptions, not just skills | Finds hidden gems the JD explicitly describes |
    | ⏰ **Temporal AI Credibility** | Pre-2023 AI work weighted 1.5x | JD: "understood retrieval before it was fashionable" |
    | 🏢 **Product Company Score** | Rewards Swiggy/Flipkart/Razorpay etc. | JD: "4-5 yrs at product companies, not services" |
    | 🚨 **Honeypot Filter** | 5-rule impossible-profile detection | >10% honeypots in top-100 = disqualification |
    | 📊 **Hirability Composite** | 8 behavioral signals → hire probability | A great candidate who doesn't respond = 0 value |
    | 💬 **Factual Reasoning** | Rank-aware, JD-referenced, no templates | Stage 4 manual review: 6 specific checks |
    """)
