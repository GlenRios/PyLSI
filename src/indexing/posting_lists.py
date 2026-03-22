"""
indexing/posting_lists.py

Inverted index built on top of posting lists.

Structure:
    index = {
        term: {
            "df":       int,               # document frequency
            "postings": {doc_id: tf, ...}  # term frequency per document
        }
    }

Persistence:
    index matrix  -> data/processed/tfidf_matrix.npz   (scipy sparse)
    index vocab   -> data/processed/index.json          (terms, doc_ids, doc_lengths)
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import PROCESSED_DIR


# -- Data structures ----------------------------------------------------------

class PostingList:
    """
    Posting list for a single term.

    Attributes:
        df:       Document frequency (number of docs containing the term).
        postings: Dict mapping doc_id -> raw term frequency (tf).
    """

    __slots__ = ("df", "postings")

    def __init__(self) -> None:
        self.df:       int            = 0
        self.postings: dict[int, int] = {}

    def add(self, doc_id: int, tf: int) -> None:
        """Inserts or updates the posting for doc_id."""
        if doc_id not in self.postings:
            self.df += 1
        self.postings[doc_id] = tf

    def __repr__(self) -> str:
        return f"PostingList(df={self.df}, docs={list(self.postings.keys())[:5]}...)"


class InvertedIndex:
    """
    Inverted index mapping terms to their posting lists.

    Also stores:
        vocab       : sorted list of all terms (row order in the matrix)
        doc_ids     : sorted list of all doc_ids (column order in the matrix)
        doc_lengths : dict mapping doc_id -> number of tokens
    """

    # Paths for persistence
    INDEX_PATH  = PROCESSED_DIR / "index.json"
    MATRIX_PATH = PROCESSED_DIR / "tfidf_matrix.npz"

    def __init__(self) -> None:
        self._index:      dict[str, PostingList] = {}
        self.vocab:       list[str]              = []
        self.doc_ids:     list[int]              = []
        self.doc_lengths: dict[int, int]         = {}

    # -- Build ----------------------------------------------------------------

    def build(self, processed_docs: list[dict]) -> "InvertedIndex":
        """
        Builds the inverted index from a list of preprocessed documents.

        Args:
            processed_docs: List of dicts with 'id' and 'tokens' fields.

        Returns:
            self (for chaining).
        """
        raw: dict[str, PostingList] = defaultdict(PostingList)

        for doc in processed_docs:
            doc_id = doc["id"]
            tokens = doc["tokens"]
            self.doc_lengths[doc_id] = len(tokens)

            tf_counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf_counts[token] += 1

            for term, tf in tf_counts.items():
                raw[term].add(doc_id, tf)

        self._index  = dict(raw)
        self.vocab   = sorted(self._index.keys())
        self.doc_ids = sorted(self.doc_lengths.keys())
        return self

    # -- Persistence ----------------------------------------------------------

    def save(self) -> None:
        """
        Saves the index metadata to disk.
        The TF-IDF matrix is saved separately via save_matrix().

        Output: data/processed/index.json
        """
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        payload = {
            "vocab":       self.vocab,
            "doc_ids":     self.doc_ids,
            "doc_lengths": {str(k): v for k, v in self.doc_lengths.items()},
            "index": {
                term: {
                    "df":       pl.df,
                    "postings": {str(k): v for k, v in pl.postings.items()},
                }
                for term, pl in self._index.items()
            },
        }

        with open(self.INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        print(f"  [SAVED] index       -> {self.INDEX_PATH}")

    @classmethod
    def load(cls) -> "InvertedIndex":
        """
        Loads a previously saved index from disk.

        Returns:
            A fully reconstructed InvertedIndex.
        """
        if not cls.INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Index not found at {cls.INDEX_PATH}. Run the pipeline first."
            )

        with open(cls.INDEX_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)

        idx             = cls()
        idx.vocab       = payload["vocab"]
        idx.doc_ids     = payload["doc_ids"]
        idx.doc_lengths = {int(k): v for k, v in payload["doc_lengths"].items()}

        for term, entry in payload["index"].items():
            pl = PostingList()
            pl.df       = entry["df"]
            pl.postings = {int(k): v for k, v in entry["postings"].items()}
            idx._index[term] = pl

        return idx

    # -- Lookup ---------------------------------------------------------------

    def __contains__(self, term: str) -> bool:
        return term in self._index

    def __getitem__(self, term: str) -> PostingList:
        return self._index[term]

    def get(self, term: str) -> PostingList | None:
        return self._index.get(term)

    def df(self, term: str) -> int:
        """Returns document frequency of a term (0 if not in index)."""
        pl = self._index.get(term)
        return pl.df if pl else 0

    def tf(self, term: str, doc_id: int) -> int:
        """Returns raw term frequency of a term in a document (0 if absent)."""
        pl = self._index.get(term)
        return pl.postings.get(doc_id, 0) if pl else 0

    # -- Stats ----------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def num_docs(self) -> int:
        return len(self.doc_ids)

    def stats(self) -> dict:
        """Returns basic index statistics."""
        dfs = [pl.df for pl in self._index.values()]
        return {
            "vocab_size":     self.vocab_size,
            "num_docs":       self.num_docs,
            "total_postings": sum(dfs),
            "avg_df":         round(sum(dfs) / len(dfs), 2) if dfs else 0,
            "max_df_term":    max(self._index, key=lambda t: self._index[t].df),
        }


# -- TF-IDF matrix ------------------------------------------------------------

def build_tfidf_matrix(
    index: InvertedIndex,
    sublinear_tf: bool = True,
) -> sp.csr_matrix:
    """
    Builds a sparse TF-IDF term-document matrix from the inverted index.

    Shape: (vocab_size, num_docs)
    Rows  = terms (ordered by index.vocab)
    Cols  = docs  (ordered by index.doc_ids)

    Formula:
        tf_weight  = 1 + log(tf)  if sublinear_tf else tf
        idf_weight = log(N / df) + 1
        tfidf      = tf_weight * idf_weight

    Args:
        index:        A built InvertedIndex.
        sublinear_tf: If True, applies log normalization to TF.

    Returns:
        Sparse CSR matrix of shape (vocab_size, num_docs).
    """
    N        = index.num_docs
    term2row = {term: i for i, term in enumerate(index.vocab)}
    doc2col  = {did:  j for j, did  in enumerate(index.doc_ids)}

    rows, cols, data = [], [], []

    for term, pl in index._index.items():
        idf = math.log(N / pl.df) + 1.0
        row = term2row[term]
        for doc_id, tf in pl.postings.items():
            tf_w = (1.0 + math.log(tf)) if sublinear_tf else float(tf)
            rows.append(row)
            cols.append(doc2col[doc_id])
            data.append(tf_w * idf)

    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(index.vocab_size, index.num_docs),
        dtype=np.float32,
    )


def save_matrix(matrix: sp.csr_matrix) -> None:
    """Saves the TF-IDF matrix to disk in scipy .npz format."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sp.save_npz(str(InvertedIndex.MATRIX_PATH), matrix)
    print(f"  [SAVED] tfidf matrix -> {InvertedIndex.MATRIX_PATH}")


def load_matrix() -> sp.csr_matrix:
    """Loads the TF-IDF matrix from disk."""
    if not InvertedIndex.MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Matrix not found at {InvertedIndex.MATRIX_PATH}. Run the pipeline first."
        )
    return sp.load_npz(str(InvertedIndex.MATRIX_PATH))


# -- Entry point --------------------------------------------------------------

def run_indexing(
    processed_docs: list[dict],
    sublinear_tf: bool = True,
) -> tuple[InvertedIndex, sp.csr_matrix]:
    """
    Runs the full indexing phase and persists results to disk.

    Args:
        processed_docs: Output of the preprocessing phase.
        sublinear_tf:   Apply log normalization to TF weights.

    Returns:
        Tuple (index, tfidf_matrix).
    """
    print("=" * 55)
    print("  LSI-Rank -- indexing")
    print("=" * 55 + "\n")

    print("  Building inverted index...")
    index = InvertedIndex().build(processed_docs)

    s = index.stats()
    print(f"  [OK] Vocabulary size  : {s['vocab_size']:,}")
    print(f"  [OK] Documents        : {s['num_docs']:,}")
    print(f"  [OK] Total postings   : {s['total_postings']:,}")
    print(f"  [OK] Avg doc freq     : {s['avg_df']}")
    print(f"  [OK] Most frequent    : '{s['max_df_term']}'")

    print("\n  Building TF-IDF matrix...")
    matrix = build_tfidf_matrix(index, sublinear_tf=sublinear_tf)
    print(f"  [OK] Matrix shape     : {matrix.shape}")
    print(f"  [OK] Non-zero entries : {matrix.nnz:,}")
    print(f"  [OK] Sparsity         : {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}")

    print("\n  Saving to disk...")
    index.save()
    save_matrix(matrix)

    print("\n" + "=" * 55)
    return index, matrix