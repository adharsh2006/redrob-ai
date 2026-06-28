# 🎯 JDMSCR — JD-Aware Multi-Signal Contextual Ranker

> **Redrob INDIA RUNS — Data & AI Challenge**  
> Submission for the AI-powered candidate ranking challenge.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-2.2%2B-green.svg)](https://www.sbert.net)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red.svg)](https://streamlit.io)

---

## 🧠 The Core Insight

The challenge JD for a **Senior AI Engineer (Founding Team)** has a deliberate trap: its `skills[]` array lists buzzwords like `langchain`, `crewai`, and `autogen` — but the actual description asks for engineers who *"understood retrieval before it was fashionable"* and built *real production ranking/recommendation systems*.

**JDMSCR solves this by reading career history descriptions, not just skill tags.**  
It scores candidates on *what they built and shipped*, using verb-proximity NLP, temporal AI credibility scoring, and 6-dimensional behavioral signals — surfacing "hidden gems" that keyword matchers miss.

---

## 📁 Repository Structure

```
.
├── src/                        # Core ranking engine
│   ├── rank.py                 # Main entry point — 3-stage pipeline
│   ├── feature_builder.py      # Stage 1 & 3: Fast features + final scoring
│   ├── career_nlp.py           # Stage 2: AI depth, seniority, career trajectory NLP
│   ├── semantic_matcher.py     # Stage 2 & 3: TF-IDF + MiniLM semantic matching
│   ├── honeypot_detector.py    # Probabilistic fake-profile detection
│   ├── reasoning_generator.py  # Factual, rank-aware reasoning output
│   └── jd_config.py            # Single source of truth: weights, keywords, timelines
├── sandbox/
│   └── app.py                  # Streamlit demo app (upload candidates -> see ranks)
├── submission/
│   └── submission.csv          # Final ranked Top-100 output
├── requirements.txt
└── README.md
```

---

## How It Works — 3-Stage Pipeline

The ranker processes up to **100,000 candidates** in a 3-stage funnel, operating entirely **offline on CPU** within a 5-minute budget.

```
100,000 candidates
       |
  [STAGE 1]  Fast Heuristics (< 10 seconds)
  -----------------------------------------------
  - Honeypot probability score (6 checks)
  - Fast keyword hits (AI/IR/NLP terms in summary + skills)
  - Behavioral availability score (notice period, open_to_work)
       |
  Top 5,000
       |
  [STAGE 2]  TF-IDF + Deep NLP (< 60 seconds)
  -----------------------------------------------
  - TF-IDF cosine similarity vs. JD text
  - AI Depth Score (4-level verb-proximity NLP)
  - Seniority Score (title + leadership verb detection)
  - Career Trajectory Score (progression analysis)
  - Experience Maturity Score (logistic curve on AI months)
       |
  Top 500
       |
  [STAGE 3]  MiniLM Neural Embeddings (< 30 seconds)
  -----------------------------------------------
  - all-MiniLM-L6-v2 cosine similarity vs JD embedding
  - Final weighted composite score
  - Monotone rank ordering + reasoning generation
       |
  Top 100 -> submission.csv
```

---

## 🔬 Scoring Components in Detail

### 1. Honeypot Detection (`honeypot_detector.py`)
Returns a continuous **probability (0.0–1.0)** rather than a binary flag. Six independent checks are combined:

| Check | What It Catches |
|---|---|
| Experience vs. Career duration gap | Claimed "15 years exp" but total job history = 3 years |
| Education timeline after work start | Degree end_year is years after earliest job start |
| Advanced skills with 0 months duration | "Advanced PyTorch" listed but 0 months used |
| Future employment dates | Job start_date is in the future |
| Technology timeline violations | Claims to have "built with ChatGPT" in 2018 |
| Impossible career progression | Intern → VP in ≤ 2 years |

Candidates with honeypot_prob > 0.8 score 0.0. All others receive a soft penalty (`score × (1 - hp_prob)`).

### 2. AI Depth Score — 4-Level Verb Proximity NLP (`career_nlp.py`)
For each career history entry, we check co-occurrence of **AI nouns** (rag, llm, ranking, search, recommendation, vector, embeddings…) with **action verbs at 4 depth levels**:

| Level | Verbs | Score per job |
|---|---|---|
| L4 — Leadership | architected, led, owned, spearheaded, designed | +20 |
| L3 — Production | deployed, scaled, optimized, productionized, shipped | +10 |
| L2 — Implementation | built, developed, implemented, created, engineered | +5 |
| L1 — Awareness | used, worked on, assisted, researched | +1 |

**Penalizes** tutorial-language ("chatbot", "course project", "bootcamp") and buzzword stuffing (many AI skills listed but no implementation verbs in descriptions).

### 3. Seniority Score (`career_nlp.py`)
Detects senior-level titles (`staff`, `principal`, `architect`, `lead`, `head`, `director`, `vp`) and leadership verbs (`architected`, `mentored`, `spearheaded`, `scaled`) in career descriptions.

### 4. Career Trajectory Score (`career_nlp.py`)
Evaluates title progression over time:
- Junior → Mid: +15
- Mid → Senior: +20
- Senior → Senior (sustained leadership): +10
- Flat title repeated: -5
- Junior → VP in ≤ 2 years: strong honeypot signal

### 5. Experience Maturity Score (`career_nlp.py`)
Counts cumulative months in AI/ML roles, then applies a **logistic curve** (midpoint at 48 months / 4 years) to normalize:

```
maturity = 100 / (1 + e^(-0.1 * (ai_months - 48)))
```

### 6. Semantic Matching (`semantic_matcher.py`)

**Stage 2 (TF-IDF):** Fast cosine similarity of candidate text (skills + career descriptions) vs. a curated JD target string. Keeps top 500.

**Stage 3 (MiniLM):** `all-MiniLM-L6-v2` sentence embeddings for deep semantic overlap. 384-dimensional cosine similarity. Takes ~0.01s/candidate on CPU.

### 7. Behavioral Signals (`feature_builder.py`)
From `redrob_signals`:
- **Response Rate** — recruiter_response_rate weighted 50%
- **Interview Completion Rate** — icr weighted 50%
- **Notice Period** — scored: ≤30d = 1.0, ≤60d = 0.8, ≤90d = 0.4, >90d = 0.1
- **Open to Work** — open_to_work_flag: True = 1.0, False = 0.2

### 8. Final Score Formula
```
FinalScore =
    0.35 x MiniLM Semantic Match
  + 0.25 x AI Depth (normalized)
  + 0.15 x Seniority + Career Trajectory (blended)
  + 0.10 x Experience Maturity
  + 0.10 x Behavioral Score
  + 0.05 x Availability Score

  x (1 - honeypot_probability)   <- soft penalty applied last
```

---

## 🚀 Running the Ranker

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run on the challenge dataset
```bash
python src/rank.py \
  --candidates /path/to/candidates.jsonl \
  --out submission.csv \
  --verbose
```

**Runtime (100K candidates):**
- Stage 1: ~5–10s
- Stage 2: ~30–60s
- Stage 3: ~20–30s (MiniLM on CPU)
- **Total: ≤ 90 seconds** on 8-core CPU, 16 GB RAM

**Constraints satisfied:**
- ✅ CPU only — no GPU required
- ✅ No network calls at ranking time (fully offline)
- ✅ Peak memory < 2 GB
- ✅ Runtime < 5 minutes for 100K candidates

---

## 🖥️ Streamlit Demo (`sandbox/app.py`)

Upload a sample candidates JSON file and see the full ranker in action with an interactive UI:

```bash
streamlit run sandbox/app.py
```

**Features:**
- Upload JSON / JSONL candidate files
- See top-N ranked candidates with score breakdowns
- Visual progress bars for each scoring dimension
- Honeypot flag display with specific flag reasons
- Download ranked output as CSV

---

## 📊 Submission Output

`submission/submission.csv` — Top 100 ranked candidates from the full challenge dataset.

| Column | Description |
|---|---|
| `candidate_id` | Challenge candidate identifier |
| `rank` | 1–100, strictly monotone |
| `score` | Final composite score (0.0–1.0), non-increasing |
| `reasoning` | Factual summary: exp years, semantic match %, AI depth, seniority, behavioral signals, top keywords |

**Example row:**
```
CAND_0068351,1,0.7897,"6.4y exp (Lead AI Engineer). Semantic Match: 66%. AI Depth: 45. Seniority: 40. Signals: 86% response rate, 0d notice. Keywords: General AI."
```

---

## 🛠️ Tech Stack

| Component | Library | Why |
|---|---|---|
| TF-IDF Vectorizer | `scikit-learn` | Fast, no GPU needed, solid baseline for Stage 2 |
| Neural Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | 384d, ~80MB, runs in 30s on CPU for 500 candidates |
| NLP Pattern Matching | Python `re` stdlib | Zero dependencies, extremely fast regex proximity matching |
| Demo App | `streamlit` | Rapid interactive UI for candidate inspection |
| Data pipeline | Pure Python stdlib (json, csv, heapq) | Streaming keeps memory flat regardless of input size |

---

## 💡 Key Design Decisions

**Why not use skill keywords?**  
The JD explicitly lists `langchain`, `crewai`, `autogen` — but then says *"frankly, if that's all you've done, you're not senior enough."* Matching on `skills[]` arrays would rank these candidates too high.

**Why a 3-stage funnel instead of scoring all 100K with MiniLM?**  
MiniLM takes ~0.01s per candidate. At 100K, that is ~17 minutes — violating the 5-minute constraint. The funnel reduces the MiniLM workload to 500 candidates (~5 seconds).

**Why probabilistic honeypot scoring instead of a hard filter?**  
Hard filters accidentally remove legitimate candidates with missing data. Soft penalties degrade the score proportionally — truly fraudulent profiles (prob > 0.8) are zeroed out, uncertain profiles are modestly penalized.

**Why heap-based streaming in Stage 1?**  
Keeps memory flat at O(5000) regardless of input size. A 100K candidate dataset never needs to be fully loaded into RAM simultaneously.

---

## 🤖 AI Tools Declaration

| Tool | Usage |
|---|---|
| Claude (Anthropic) | Architecture ideation, scoring formula design, code review |
| No LLM at ranking time | Ranker is 100% offline — zero API calls during execution |

---

## 📄 License

MIT License — see challenge terms for dataset usage restrictions.
