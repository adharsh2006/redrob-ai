"""
rank.py
-------
MAIN SUBMISSION ENGINE — Redrob Hackathon (V2 Architecture)
3-Stage Funnel (100K -> 5000 -> 500 -> 100)

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints:
    Runtime  : ≤ 5 minutes
    Memory   : ≤ 16 GB
    Compute  : CPU only
"""

import argparse
import csv
import json
import sys
import time
import heapq
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from feature_builder import build_fast_features, compute_final_score
from career_nlp import get_v2_career_scores
from semantic_matcher import FastSemanticMatcher, DeepSemanticMatcher
from reasoning_generator import generate_reasoning

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=str,
        default=r"D:\projects\REDROB AI\dataset\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    )
    parser.add_argument("--out", type=str, default="submission.csv")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def stream_candidates(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# ── 3-Stage Pipeline ───────────────────────────────────────────────────────────

def execute_pipeline(candidates_path: str, verbose: bool = False):
    t0 = time.time()
    
    # --- STAGE 1: 100K -> 5000 ---
    if verbose:
        print("[STAGE 1] Streaming 100K candidates for fast heuristics...")
        
    heap_5000 = []
    total = 0
    
    for candidate in stream_candidates(candidates_path):
        total += 1
        fast_features = build_fast_features(candidate)
        score = fast_features["stage1_score"]
        
        cid = candidate["candidate_id"]
        entry = (score, cid, fast_features, candidate)
        
        if len(heap_5000) < 5000:
            heapq.heappush(heap_5000, entry)
        elif score > heap_5000[0][0]:
            heapq.heapreplace(heap_5000, entry)
            
        if verbose and total % 20_000 == 0:
            print(f"  [{time.time() - t0:.1f}s] Processed {total:,} candidates")

    # Extract the Top 5000 candidates
    top_5000_candidates = [e[3] for e in heap_5000]
    top_5000_features = {e[1]: e[2] for e in heap_5000}
    
    if verbose:
        print(f"[STAGE 1 DONE] Kept {len(top_5000_candidates)} candidates. Elapsed: {time.time() - t0:.1f}s")
        print("[STAGE 2] Running TF-IDF & Deep NLP on Top 5000...")
        
    # --- STAGE 2: 5000 -> 500 ---
    fast_matcher = FastSemanticMatcher()
    tfidf_scores = fast_matcher.fit_and_score(top_5000_candidates)
    
    heap_500 = []
    
    for idx, candidate in enumerate(top_5000_candidates):
        cid = candidate["candidate_id"]
        fast_feat = top_5000_features[cid]
        tfidf_score = tfidf_scores[idx]
        
        # Deep NLP extraction
        v2_scores = get_v2_career_scores(candidate)
        
        # Intermediate Score (TF-IDF + AI Depth + Seniority)
        # Weight TF-IDF higher in Stage 2 to pull in true semantic overlap
        intermediate_score = (
            tfidf_score * 0.4 +
            (v2_scores["ai_depth"] / 100.0) * 0.3 +
            (v2_scores["seniority"] / 50.0) * 0.2 +
            fast_feat["behavioral"] * 0.1
        )
        
        # Apply honeypot penalty soft
        intermediate_score *= max(0.0, 1.0 - fast_feat["honeypot_prob"])
        
        entry = (intermediate_score, cid, fast_feat, v2_scores, candidate)
        
        if len(heap_500) < 500:
            heapq.heappush(heap_500, entry)
        elif intermediate_score > heap_500[0][0]:
            heapq.heapreplace(heap_500, entry)

    top_500_entries = list(heap_500)
    top_500_candidates_only = [e[4] for e in top_500_entries]
    
    if verbose:
        print(f"[STAGE 2 DONE] Kept {len(top_500_candidates_only)} candidates. Elapsed: {time.time() - t0:.1f}s")
        print("[STAGE 3] Running MiniLM SentenceTransformers on Top 500...")
        
    # --- STAGE 3: 500 -> 100 ---
    # Load neural embeddings
    deep_matcher = DeepSemanticMatcher()
    minilm_scores = deep_matcher.score_candidates(top_500_candidates_only)
    
    final_results = []
    
    for idx, entry in enumerate(top_500_entries):
        _, cid, fast_feat, v2_scores, candidate = entry
        minilm_sim = minilm_scores[idx]
        
        final_score = compute_final_score(fast_feat, v2_scores, minilm_sim)
        final_results.append((final_score, cid, fast_feat, v2_scores, minilm_sim, candidate))
        
    # Sort descending by final score, break ties by ascending CID
    final_results.sort(key=lambda x: (-round(x[0], 4), x[1]))
    
    top_100 = final_results[:100]
    
    if verbose:
        print(f"[STAGE 3 DONE] Final Top 100 selected. Elapsed: {time.time() - t0:.1f}s")
        
    return top_100, total

# ── Output CSV ────────────────────────────────────────────────────────────────

def write_submission(top_100: list, output_path: str, verbose: bool = False):
    output = Path(output_path)
    
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        prev_score = None
        for rank_idx, result in enumerate(top_100, start=1):
            raw_score, cid, fast_feat, v2_scores, minilm_sim, candidate = result
            
            if prev_score is not None and raw_score > prev_score:
                raw_score = prev_score
            prev_score = raw_score
            
            display_score = round(raw_score, 4)
            reasoning = generate_reasoning(candidate, fast_feat, v2_scores, minilm_sim)
            
            writer.writerow([cid, rank_idx, display_score, reasoning])
            
            if verbose and rank_idx <= 10:
                print(f"  Rank {rank_idx:3}: {cid} | score={display_score:.4f} | {candidate['profile'].get('current_title','')[:30]}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t_start = time.time()
    
    print("=" * 60)
    print("  JDMSCR V2 — 3-Stage Semantic Ranker")
    print("============================================================")
    
    top_100, total = execute_pipeline(args.candidates, verbose=args.verbose)
    write_submission(top_100, args.out, verbose=args.verbose)
    
    t_total = time.time() - t_start
    print(f"\n[OK] Pipeline completed in {t_total:.1f}s")
    print(f"[OK] Output saved to: {args.out}")

if __name__ == "__main__":
    main()
