"""
preprocessing/pipeline.py

Text cleaning and normalization pipeline.

Dependencies:
    - scikit-learn : English stopword list
    - scipy        : used in later phases (indexing, LSI)

Steps applied to every document and query:
    1. Lowercase
    2. Strip numbers and punctuation
    3. Tokenize (regex-based, no external dependency)
    4. Remove stopwords (sklearn's built-in English list)
    5. Drop short tokens (below min_token_length)
    6. Stem (Porter algorithm, built-in implementation)
"""

import re
import sys
from pathlib import Path

from src.preprocessing.stopwords import ENGLISH as ENGLISH_STOP_WORDS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import PREPROCESSING
from src.preprocessing.stemmer import stem as porter_stem


# -- Module-level singletons --------------------------------------------------
_stopwords = set(ENGLISH_STOP_WORDS)


# -- Individual steps ---------------------------------------------------------

def lowercase(text: str) -> str:
    """Converts text to lowercase."""
    return text.lower()


def remove_special_chars(text: str) -> str:
    """Removes numbers, punctuation, and non-alphabetic characters."""
    return re.sub(r"[^a-z\s]", " ", text)


def tokenize(text: str) -> list[str]:
    """Splits text into tokens using a regex word boundary."""
    return re.findall(r"[a-z]+", text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Removes tokens that appear in sklearn's English stopword list."""
    return [t for t in tokens if t not in _stopwords]


def filter_short(tokens: list[str], min_length: int) -> list[str]:
    """Drops tokens shorter than min_length characters."""
    return [t for t in tokens if len(t) >= min_length]


def apply_stemming(tokens: list[str]) -> list[str]:
    """Applies Porter stemming."""
    return [porter_stem(t) for t in tokens]


# -- Full pipeline ------------------------------------------------------------

def preprocess(
    text: str,
    remove_stops: bool  = PREPROCESSING["remove_stopwords"],
    min_length: int     = PREPROCESSING["min_token_length"],
    stemmer: str | None = PREPROCESSING["stemmer"],
) -> list[str]:
    """
    Runs the full preprocessing pipeline on a raw text string.

    Args:
        text:         Raw input text.
        remove_stops: Whether to remove stopwords.
        min_length:   Minimum token length to keep.
        stemmer:      Stemmer to use: "porter" or None.

    Returns:
        List of cleaned, normalized tokens.

    Example:
        >>> preprocess("The aerodynamic properties of lifting surfaces")
        ['aerodynam', 'properti', 'lift', 'surfac']
    """
    text   = lowercase(text)
    text   = remove_special_chars(text)
    tokens = tokenize(text)

    if remove_stops:
        tokens = remove_stopwords(tokens)

    tokens = filter_short(tokens, min_length)

    if stemmer == "porter":
        tokens = apply_stemming(tokens)

    return tokens


def preprocess_document(doc: dict) -> dict:
    """
    Preprocesses a parsed CRAN document.
    Concatenates title and abstract, then runs the pipeline.

    Args:
        doc: Dict with keys 'id', 't' (title), 'w' (abstract).

    Returns:
        Original dict extended with 'tokens' (list[str]).
    """
    raw_text = f"{doc.get('t', '')} {doc.get('w', '')}"
    return {**doc, "tokens": preprocess(raw_text)}


def preprocess_query(query: dict) -> dict:
    """
    Preprocesses a parsed CRAN query.

    Args:
        query: Dict with keys 'id', 'w' (query text).

    Returns:
        Original dict extended with 'tokens' (list[str]).
    """
    return {**query, "tokens": preprocess(query.get("w", ""))}


# -- Batch processing ---------------------------------------------------------

def preprocess_corpus(docs: list[dict]) -> list[dict]:
    """
    Preprocesses a list of documents. Logs progress every 200 docs.

    Args:
        docs: List of parsed document dicts.

    Returns:
        List of dicts, each extended with a 'tokens' field.
    """
    processed = []
    for i, doc in enumerate(docs, 1):
        processed.append(preprocess_document(doc))
        if i % 200 == 0 or i == len(docs):
            print(f"  [PREPROCESSING] {i}/{len(docs)} documents processed")
    return processed


def preprocess_queries(queries: list[dict]) -> list[dict]:
    """
    Preprocesses a list of queries.

    Args:
        queries: List of parsed query dicts.

    Returns:
        List of dicts, each extended with a 'tokens' field.
    """
    return [preprocess_query(q) for q in queries]


# -- Stats --------------------------------------------------------------------

def vocabulary_stats(processed_docs: list[dict]) -> dict:
    """
    Computes basic vocabulary statistics over the preprocessed corpus.

    Returns:
        Dict with keys: vocab_size, total_tokens, avg_tokens_per_doc,
        min_tokens, max_tokens.
    """
    all_tokens = [t for doc in processed_docs for t in doc["tokens"]]
    lengths    = [len(doc["tokens"]) for doc in processed_docs]

    return {
        "vocab_size":         len(set(all_tokens)),
        "total_tokens":       len(all_tokens),
        "avg_tokens_per_doc": round(len(all_tokens) / len(processed_docs), 1),
        "min_tokens":         min(lengths),
        "max_tokens":         max(lengths),
    }


# -- Entry point --------------------------------------------------------------

def run_preprocessing(docs: list[dict], queries: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Runs the full preprocessing phase.

    Args:
        docs:    List of parsed document dicts (from ingestion).
        queries: List of parsed query dicts (from ingestion).

    Returns:
        Tuple (processed_docs, processed_queries).
    """
    print("=" * 55)
    print("  LSI-Rank -- preprocessing")
    print("=" * 55 + "\n")

    processed_docs    = preprocess_corpus(docs)
    processed_queries = preprocess_queries(queries)

    save_processed(processed_docs, processed_queries)

    stats = vocabulary_stats(processed_docs)

    print("\n-- Corpus stats -------------------------------------")
    print(f"  Vocabulary size    : {stats['vocab_size']:,}")
    print(f"  Total tokens       : {stats['total_tokens']:,}")
    print(f"  Avg tokens/doc     : {stats['avg_tokens_per_doc']}")
    print(f"  Min tokens/doc     : {stats['min_tokens']}")
    print(f"  Max tokens/doc     : {stats['max_tokens']}")
    print("=" * 55)

    return processed_docs, processed_queries


def save_processed(
    processed_docs: list[dict],
    processed_queries: list[dict],
) -> None:
    """
    Persists preprocessed tokens back into the JSONL files on disk.
    This allows diagnose.py and other tools to load tokenized data directly.
    """
    import json, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config.settings import PARSED_DOCS_PATH, PARSED_QUERIES_PATH

    with open(PARSED_DOCS_PATH, "w", encoding="utf-8") as f:
        for doc in processed_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with open(PARSED_QUERIES_PATH, "w", encoding="utf-8") as f:
        for q in processed_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"  [SAVED] tokenized docs    -> {PARSED_DOCS_PATH}")
    print(f"  [SAVED] tokenized queries -> {PARSED_QUERIES_PATH}")