"""
evaluation/metrics.py

Standard IR evaluation metrics for LSI-Rank.

Metrics implemented:
    - Precision@K   : fraction of top-K results that are relevant
    - Recall@K      : fraction of all relevant docs found in top-K
    - F1@K          : harmonic mean of Precision@K and Recall@K
    - AP            : Average Precision for a single query
    - MAP           : Mean Average Precision across all queries
    - NDCG@K        : Normalized Discounted Cumulative Gain

All functions operate on:
    retrieved : list of doc_ids in ranked order (best first)
    relevant  : set of doc_ids that are truly relevant for this query
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import EVALUATION, METRICS_DIR
from src.indexing.posting_lists import InvertedIndex
from src.models.lsi import LSIModel
from src.retrieval.engine import retrieve_from_tokens


# -- Single-query metrics -----------------------------------------------------

def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Fraction of top-K retrieved documents that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for d in top_k if d in relevant) / k


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Fraction of all relevant documents found in the top-K results."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for d in top_k if d in relevant) / len(relevant)


def f1_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision(retrieved: list[int], relevant: set[int]) -> float:
    """
    Average Precision (AP) for a single query.

    Computes the mean of Precision@K values at each position where
    a relevant document is found.

    Returns 0.0 if there are no relevant documents.
    """
    if not relevant:
        return 0.0

    hits, ap = 0, 0.0
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            hits += 1
            ap   += hits / i

    return ap / len(relevant)


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    Uses binary relevance: 1 if relevant, 0 otherwise.
    Ideal DCG assumes all relevant docs appear at the top.
    """
    def dcg(ranked: list[int], rel: set[int], cutoff: int) -> float:
        return sum(
            1.0 / math.log2(i + 2)
            for i, doc_id in enumerate(ranked[:cutoff])
            if doc_id in rel
        )

    actual_dcg = dcg(retrieved, relevant, k)
    # Ideal: all relevant docs ranked first
    ideal_list = list(relevant) + [d for d in retrieved if d not in relevant]
    ideal_dcg  = dcg(ideal_list, relevant, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# -- Corpus-level evaluation --------------------------------------------------

def evaluate_all(
    processed_queries: list[dict],
    qrels_dict: dict[int, dict[int, bool]],
    index: InvertedIndex,
    lsi_model: LSIModel,
    top_k: int = 1400,
    cutoffs: list[int] = EVALUATION["cutoffs"],
) -> dict:
    """
    Evaluates the LSI model over all queries with ground-truth relevance judgments.

    Args:
        processed_queries : List of preprocessed query dicts (with 'id', 'tokens').
        qrels_dict        : {query_id: {doc_id: relevant (bool)}}
        index             : Built InvertedIndex.
        lsi_model         : Fitted LSIModel.
        top_k             : Number of documents to retrieve per query.
        cutoffs           : List of K values for Precision/Recall/NDCG@K.

    Returns:
        Dict with per-query results and aggregated metrics.
    """
    per_query  : list[dict] = []
    ap_scores  : list[float] = []

    # Accumulators for each cutoff
    p_sums  = {k: 0.0 for k in cutoffs}
    r_sums  = {k: 0.0 for k in cutoffs}
    f1_sums = {k: 0.0 for k in cutoffs}
    ndcg_sums = {k: 0.0 for k in cutoffs}

    evaluated = 0

    for query in processed_queries:
        qid     = query["id"]
        tokens  = query["tokens"]
        rel_map = qrels_dict.get(qid, {})
        relevant = {doc_id for doc_id, is_rel in rel_map.items() if is_rel}

        # Skip queries with no relevant documents
        if not relevant:
            continue

        results   = retrieve_from_tokens(tokens, index, lsi_model, top_k=top_k)
        retrieved = [doc_id for doc_id, _ in results]

        ap = average_precision(retrieved, relevant)
        ap_scores.append(ap)

        query_metrics: dict = {"query_id": qid, "ap": round(ap, 4), "num_relevant": len(relevant)}

        for k in cutoffs:
            p  = precision_at_k(retrieved, relevant, k)
            r  = recall_at_k(retrieved, relevant, k)
            f1 = f1_at_k(retrieved, relevant, k)
            nd = ndcg_at_k(retrieved, relevant, k)

            p_sums[k]    += p
            r_sums[k]    += r
            f1_sums[k]   += f1
            ndcg_sums[k] += nd

            query_metrics[f"p@{k}"]    = round(p,  4)
            query_metrics[f"r@{k}"]    = round(r,  4)
            query_metrics[f"f1@{k}"]   = round(f1, 4)
            query_metrics[f"ndcg@{k}"] = round(nd, 4)

        per_query.append(query_metrics)
        evaluated += 1

        if evaluated % 50 == 0:
            print(f"  [EVAL] {evaluated}/{len(processed_queries)} queries evaluated...")

    # Aggregate
    n = evaluated
    aggregated: dict = {
        "num_queries_evaluated": n,
        "MAP": round(np.mean(ap_scores), 4),
    }
    for k in cutoffs:
        aggregated[f"P@{k}"]    = round(p_sums[k]    / n, 4)
        aggregated[f"R@{k}"]    = round(r_sums[k]    / n, 4)
        aggregated[f"F1@{k}"]   = round(f1_sums[k]   / n, 4)
        aggregated[f"NDCG@{k}"] = round(ndcg_sums[k] / n, 4)

    return {"aggregated": aggregated, "per_query": per_query}


# -- Persistence --------------------------------------------------------------

def save_results(results: dict, k: int) -> Path:
    """Saves evaluation results to results/metrics/eval_k{k}.json."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"eval_k{k}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  [SAVED] results -> {path}")
    return path


# -- Entry point --------------------------------------------------------------

def run_evaluation(
    processed_queries: list[dict],
    qrels_dict: dict[int, dict[int, bool]],
    index: InvertedIndex,
    lsi_model: LSIModel,
) -> dict:
    """
    Runs the full evaluation phase.

    Args:
        processed_queries : Preprocessed queries from Phase 2.
        qrels_dict        : Relevance judgments from ingestion.
        index             : Built InvertedIndex from Phase 3.
        lsi_model         : Fitted LSIModel from Phase 4.

    Returns:
        Dict with aggregated and per-query metrics.
    """
    print("=" * 55)
    print("  LSI-Rank -- evaluation")
    print("=" * 55 + "\n")

    results = evaluate_all(
        processed_queries, qrels_dict, index, lsi_model
    )

    agg = results["aggregated"]
    n   = agg["num_queries_evaluated"]

    print(f"\n-- Results (k={lsi_model.k}, {n} queries) ----------")
    print(f"  {'Metric':<12} {'Value':>8}")
    print(f"  {'-'*12} {'-'*8}")
    print(f"  {'MAP':<12} {agg['MAP']:>8.4f}")
    for cutoff in EVALUATION["cutoffs"]:
        print(f"  {'P@'+str(cutoff):<12} {agg[f'P@{cutoff}']:>8.4f}")
        print(f"  {'R@'+str(cutoff):<12} {agg[f'R@{cutoff}']:>8.4f}")
        print(f"  {'NDCG@'+str(cutoff):<12} {agg[f'NDCG@{cutoff}']:>8.4f}")

    save_results(results, k=lsi_model.k)
    print("\n" + "=" * 55)
    return results