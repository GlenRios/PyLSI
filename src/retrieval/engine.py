"""
retrieval/engine.py

Retrieval engine for LSI-Rank.

Pipeline for a single query:
    1. Preprocess query tokens (same pipeline as documents)
    2. Build a TF-IDF vector in the original term space
    3. Project into the LSI latent space: q_lsi = q_tfidf @ Uk * Sk^-1
    4. Compute cosine similarity against all document vectors (Vk)
    5. Return ranked list of (doc_id, score) pairs
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import RETRIEVAL
from src.indexing.posting_lists import InvertedIndex
from src.models.lsi import LSIModel
from src.preprocessing.pipeline import preprocess


# -- Query vectorization ------------------------------------------------------

def vectorize_query(
    tokens: list[str],
    index: InvertedIndex,
    sublinear_tf: bool = True,
) -> np.ndarray:
    """
    Converts preprocessed query tokens into a TF-IDF vector
    aligned with the index vocabulary.

    Args:
        tokens:       Preprocessed query tokens.
        index:        The built InvertedIndex (provides vocab and IDF weights).
        sublinear_tf: Apply log normalization to TF, same as during indexing.

    Returns:
        Dense vector of shape (vocab_size,).
    """
    import math

    N        = index.num_docs
    term2row = {term: i for i, term in enumerate(index.vocab)}
    vector   = np.zeros(index.vocab_size, dtype=np.float64)

    tf_counts: dict[str, int] = {}
    for token in tokens:
        if token in index:
            tf_counts[token] = tf_counts.get(token, 0) + 1

    for term, tf in tf_counts.items():
        row  = term2row[term]
        tf_w = (1.0 + math.log(tf)) if sublinear_tf else float(tf)
        idf  = math.log(N / index.df(term)) + 1.0
        vector[row] = tf_w * idf

    return vector


# -- Cosine similarity --------------------------------------------------------

def cosine_similarity(
    query_vec: np.ndarray,
    doc_matrix: np.ndarray,
) -> np.ndarray:
    """
    Computes cosine similarity between a query vector and all document vectors.

    Args:
        query_vec:  Dense vector of shape (k,).
        doc_matrix: Dense matrix of shape (num_docs, k).

    Returns:
        Array of shape (num_docs,) with similarity scores in [-1, 1].
        Returns zeros if the query vector has zero norm.
    """
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return np.zeros(doc_matrix.shape[0])

    d_norms = np.linalg.norm(doc_matrix, axis=1)
    d_norms = np.where(d_norms == 0, 1e-10, d_norms)

    return (doc_matrix @ query_vec) / (d_norms * q_norm)


# -- Retrieval ----------------------------------------------------------------

def retrieve(
    query_text: str,
    index: InvertedIndex,
    lsi_model: LSIModel,
    top_k: int = RETRIEVAL["top_k"],
) -> list[tuple[int, float]]:
    """
    Retrieves the top-k most relevant documents for a raw query string.

    Args:
        query_text: Raw query string (preprocessed internally).
        index:      Built InvertedIndex.
        lsi_model:  Fitted LSIModel.
        top_k:      Number of results to return.

    Returns:
        List of (doc_id, score) tuples sorted by descending score.
    """
    tokens      = preprocess(query_text)
    query_tfidf = vectorize_query(tokens, index)
    query_lsi   = lsi_model.project_query(query_tfidf)
    scores      = cosine_similarity(query_lsi, lsi_model.Vk)

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(index.doc_ids[i], float(scores[i])) for i in top_indices]


def retrieve_from_tokens(
    tokens: list[str],
    index: InvertedIndex,
    lsi_model: LSIModel,
    top_k: int = RETRIEVAL["top_k"],
) -> list[tuple[int, float]]:
    """
    Same as retrieve() but accepts already-preprocessed tokens.
    Used by the evaluation phase to avoid re-preprocessing.

    Args:
        tokens:    Already preprocessed query tokens.
        index:     Built InvertedIndex.
        lsi_model: Fitted LSIModel.
        top_k:     Number of results to return.

    Returns:
        List of (doc_id, score) tuples sorted by descending score.
    """
    query_tfidf = vectorize_query(tokens, index)
    query_lsi   = lsi_model.project_query(query_tfidf)
    scores      = cosine_similarity(query_lsi, lsi_model.Vk)

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(index.doc_ids[i], float(scores[i])) for i in top_indices]


# -- Entry point --------------------------------------------------------------

def run_retrieval_demo(
    processed_queries: list[dict],
    index: InvertedIndex,
    lsi_model: LSIModel,
    n_examples: int = 3,
) -> None:
    """
    Runs a quick demo showing results for the first n_examples queries.

    Args:
        processed_queries: Output of the preprocessing phase.
        index:             Built InvertedIndex.
        lsi_model:         Fitted LSIModel.
        n_examples:        Number of queries to display.
    """
    print("=" * 55)
    print("  LSI-Rank -- retrieval demo")
    print("=" * 55)

    for query in processed_queries[:n_examples]:
        results = retrieve_from_tokens(
            query["tokens"], index, lsi_model, top_k=5
        )
        print(f"\n  Query {query['id']:>3} | {query.get('w', '')[:60]}...")
        print(f"  {'Doc ID':<10} {'Score':>8}")
        print(f"  {'-'*10} {'-'*8}")
        for doc_id, score in results:
            print(f"  {doc_id:<10} {score:>8.4f}")

    print("\n" + "=" * 55)