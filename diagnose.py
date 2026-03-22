"""
diagnose.py

Diagnostic tool to identify why MAP is low.
Run this from the project root after running main.py at least once.

    python diagnose.py

Checks:
    1. Data integrity  -- doc/query/qrel counts and ID ranges
    2. Qrel analysis   -- relevance distribution, queries without judgments
    3. Token coverage  -- fraction of query tokens that exist in the index
    4. Query vectors   -- how sparse/empty query TF-IDF vectors are
    5. Sample retrieval -- top-10 results for 3 queries vs ground truth
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import PROCESSED_DIR, QUERIES_DIR, QRELS_DIR
from src.ingestion.parser import load_documents, load_queries, load_qrels
from src.indexing.posting_lists import InvertedIndex, load_matrix
from src.models.lsi import LSIModel
from src.preprocessing.pipeline import preprocess
from src.retrieval.engine import vectorize_query, retrieve_from_tokens


def sep(title=""):
    print(f"\n{'='*55}")
    if title:
        print(f"  {title}")
        print("=" * 55)


# -- 1. Data integrity --------------------------------------------------------

def check_data():
    sep("1. Data integrity")
    docs    = load_documents()
    queries = load_queries()
    qrels   = load_qrels()

    doc_ids   = [d["id"] for d in docs]
    query_ids = [q["id"] for q in queries]

    print(f"  Documents : {len(docs)} | IDs {min(doc_ids)}..{max(doc_ids)}")
    print(f"  Queries   : {len(queries)} | IDs {min(query_ids)}..{max(query_ids)}")
    print(f"  Qrels     : {len(qrels)}")

    # Check for duplicate IDs
    if len(set(doc_ids)) != len(doc_ids):
        print("  [WARN] Duplicate document IDs detected!")
    if len(set(query_ids)) != len(query_ids):
        print("  [WARN] Duplicate query IDs detected!")

    # Check token lengths in first few docs
    print(f"\n  Sample doc tokens  : {docs[0].get('tokens', 'NOT PREPROCESSED')[:8]}")
    print(f"  Sample query tokens: {queries[0].get('tokens', 'NOT PREPROCESSED')[:8]}")

    return docs, queries, qrels


# -- 2. Qrel analysis ---------------------------------------------------------

def check_qrels(qrels):
    sep("2. Qrel analysis")

    rel_counts   = Counter(q["relevance"] for q in qrels)
    qrel_by_qid  = defaultdict(list)
    for q in qrels:
        qrel_by_qid[q["query_id"]].append(q)

    print(f"  Relevance distribution : {dict(sorted(rel_counts.items()))}")
    print(f"  Queries with any qrels : {len(qrel_by_qid)}/225")

    # How many queries have at least one relevant doc (relevance > 0)
    with_relevant = {
        qid for qid, entries in qrel_by_qid.items()
        if any(e["relevance"] > 0 for e in entries)
    }
    print(f"  Queries with rel > 0   : {len(with_relevant)}")

    rel_per_query = [
        sum(1 for e in entries if e["relevance"] > 0)
        for entries in qrel_by_qid.values()
    ]
    print(f"  Avg relevant/query     : {np.mean(rel_per_query):.1f}")
    print(f"  Min / Max relevant     : {min(rel_per_query)} / {max(rel_per_query)}")

    return qrel_by_qid


# -- 3. Token coverage --------------------------------------------------------

def check_token_coverage(queries, index):
    sep("3. Query token coverage in index")

    coverages = []
    zero_coverage = 0

    for q in queries:
        tokens = q.get("tokens", [])
        if not tokens:
            continue
        in_index = sum(1 for t in tokens if t in index)
        cov = in_index / len(tokens)
        coverages.append(cov)
        if cov == 0:
            zero_coverage += 1

    print(f"  Avg token coverage    : {np.mean(coverages):.2%}")
    print(f"  Median coverage       : {np.median(coverages):.2%}")
    print(f"  Queries with 0% cover : {zero_coverage}/{len(coverages)}")
    print(f"  Queries with <25% cov : {sum(1 for c in coverages if c < 0.25)}")

    # Show worst queries
    worst = sorted(zip(coverages, [q['id'] for q in queries if q.get('tokens')]))[:5]
    print(f"\n  Lowest coverage queries:")
    for cov, qid in worst:
        q = next(x for x in queries if x['id'] == qid)
        print(f"    Query {qid:>3}: {cov:.0%} | tokens: {q.get('tokens', [])[:6]}")

    return coverages


# -- 4. Query vector sparsity -------------------------------------------------

def check_query_vectors(queries, index):
    sep("4. Query TF-IDF vector sparsity")

    nonzero_counts = []
    for q in queries:
        tokens = q.get("tokens", [])
        vec    = vectorize_query(tokens, index)
        nonzero_counts.append(int(np.count_nonzero(vec)))

    zero_vecs = sum(1 for n in nonzero_counts if n == 0)
    print(f"  Vocab size             : {index.vocab_size:,}")
    print(f"  Avg non-zero per query : {np.mean(nonzero_counts):.1f}")
    print(f"  Queries with zero vec  : {zero_vecs}/{len(nonzero_counts)}")
    print(f"  Min / Max non-zero     : {min(nonzero_counts)} / {max(nonzero_counts)}")

    return nonzero_counts


# -- 5. Sample retrieval ------------------------------------------------------

def check_sample_retrieval(queries, qrel_by_qid, index, lsi_model, n=3):
    sep("5. Sample retrieval (first 3 queries with qrels)")

    evaluated = 0
    for q in queries:
        qid    = q["id"]
        tokens = q.get("tokens", [])
        if qid not in qrel_by_qid or not tokens:
            continue

        relevant = {e["doc_id"] for e in qrel_by_qid[qid] if e["relevance"] > 0}
        results  = retrieve_from_tokens(tokens, index, lsi_model, top_k=20)

        print(f"\n  Query {qid}: '{q.get('w','')[:60]}...'")
        print(f"  Tokens   : {tokens[:8]}")
        print(f"  Relevant : {len(relevant)} docs | e.g. {sorted(relevant)[:5]}")
        print(f"  {'Rank':<5} {'DocID':<8} {'Score':>8}  {'Relevant?'}")
        print(f"  {'-'*5} {'-'*8} {'-'*8}  {'-'*9}")
        for rank, (doc_id, score) in enumerate(results[:10], 1):
            tag = "✓" if doc_id in relevant else ""
            print(f"  {rank:<5} {doc_id:<8} {score:>8.4f}  {tag}")

        evaluated += 1
        if evaluated >= n:
            break


# -- Main ---------------------------------------------------------------------

def main():
    print("LSI-Rank — Diagnostic Report")

    docs, queries, qrels = check_data()

    # Need preprocessed tokens — load from processed files if available
    proc_queries_path = QUERIES_DIR / "queries.jsonl"
    if proc_queries_path.exists():
        proc_queries = [json.loads(l) for l in proc_queries_path.read_text().splitlines()]
        if proc_queries[0].get("tokens"):
            queries = proc_queries

    proc_docs_path = PROCESSED_DIR / "documents.jsonl"
    if proc_docs_path.exists():
        proc_docs = [json.loads(l) for l in proc_docs_path.read_text().splitlines()]
        if proc_docs[0].get("tokens"):
            docs = proc_docs

    qrel_by_qid = check_qrels(qrels)

    # Load index
    try:
        index = InvertedIndex.load()
        print(f"\n  Index loaded: {index.vocab_size:,} terms, {index.num_docs:,} docs")
    except FileNotFoundError as e:
        print(f"\n  [ERROR] {e}")
        return

    check_token_coverage(queries, index)
    check_query_vectors(queries, index)

    # Load LSI model (try k=150 then k=100)
    lsi_model = None
    for k in [150, 100, 50]:
        try:
            lsi_model = LSIModel.load(k=k)
            print(f"\n  LSI model loaded: k={k}")
            print(f"  Variance explained: {lsi_model.cumulative_variance()[-1]:.2%}")
            break
        except FileNotFoundError:
            continue

    if lsi_model is None:
        print("\n  [ERROR] No LSI model found on disk. Run main.py first.")
        return

    check_sample_retrieval(queries, qrel_by_qid, index, lsi_model)

    sep("Done")
    print("  Key things to look for:")
    print("  - Token coverage < 50% → preprocessing mismatch")
    print("  - Many zero-vector queries → OOV terms")
    print("  - Good scores for irrelevant docs → LSI space not separating topics")


if __name__ == "__main__":
    main()