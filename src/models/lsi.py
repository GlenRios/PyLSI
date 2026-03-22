"""
models/lsi.py

Latent Semantic Indexing (LSI) model via truncated SVD.

Given a TF-IDF matrix M of shape (vocab_size, num_docs):

    M ≈ Uk · Sk · Vk^T

Where:
    Uk  : (vocab_size, k)  -- term vectors in latent space
    Sk  : (k, k)           -- diagonal matrix of singular values
    Vk  : (num_docs, k)    -- document vectors in latent space

Query projection into latent space:
    q_lsi = q_tfidf · Uk · Sk^-1

Improvements applied before SVD:
    - L2 column normalization: each document column is normalized to unit
      length so that document length does not dominate the decomposition.

Persistence:
    data/processed/lsi_Uk_k{k}.npy
    data/processed/lsi_Sk_k{k}.npy
    data/processed/lsi_Vk_k{k}.npy
    data/processed/lsi_meta_k{k}.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import PROCESSED_DIR, LSI


# -- Paths --------------------------------------------------------------------

def _paths(k: int) -> dict[str, Path]:
    return {
        "Uk":   PROCESSED_DIR / f"lsi_Uk_k{k}.npy",
        "Sk":   PROCESSED_DIR / f"lsi_Sk_k{k}.npy",
        "Vk":   PROCESSED_DIR / f"lsi_Vk_k{k}.npy",
        "meta": PROCESSED_DIR / f"lsi_meta_k{k}.json",
    }


# -- Normalization ------------------------------------------------------------

def normalize_columns(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """
    L2-normalizes each column (document) of a sparse matrix.

    This prevents long documents from dominating the SVD decomposition
    and is standard practice in LSI implementations.

    Args:
        matrix: Sparse TF-IDF matrix of shape (vocab_size, num_docs).

    Returns:
        Column-normalized sparse matrix of the same shape.
    """
    # Compute L2 norm of each column
    col_norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=0))).flatten()
    # Avoid division by zero
    col_norms = np.where(col_norms == 0, 1.0, col_norms)
    # Normalize: divide each column by its norm
    norm_diag = sp.diags(1.0 / col_norms)
    return matrix @ norm_diag


# -- Model --------------------------------------------------------------------

class LSIModel:
    """
    LSI model wrapping the truncated SVD decomposition.

    Attributes:
        k     : number of latent dimensions
        Uk    : (vocab_size, k) term matrix
        Sk    : (k,)            singular values
        Vk    : (num_docs, k)   document matrix in latent space
        Sk_inv: (k,)            inverse singular values (for query projection)
    """

    def __init__(self, k: int = LSI["k_default"]) -> None:
        self.k       = k
        self.Uk:     np.ndarray | None = None
        self.Sk:     np.ndarray | None = None
        self.Vk:     np.ndarray | None = None
        self.Sk_inv: np.ndarray | None = None

    # -- Fit ------------------------------------------------------------------

    def fit(self, tfidf_matrix: sp.csr_matrix) -> "LSIModel":
        """
        Normalizes the TF-IDF matrix and computes the truncated SVD.

        scipy.sparse.linalg.svds returns singular values in ascending order,
        so we reverse them to get descending order (largest variance first).

        Args:
            tfidf_matrix: Sparse TF-IDF matrix of shape (vocab_size, num_docs).

        Returns:
            self (for chaining).
        """
        if self.k >= min(tfidf_matrix.shape):
            raise ValueError(
                f"k={self.k} must be smaller than min(vocab_size, num_docs)="
                f"{min(tfidf_matrix.shape)}"
            )

        print(f"  Normalizing columns...")
        matrix = normalize_columns(tfidf_matrix)

        print(f"  Computing SVD (k={self.k})...")
        Uk, Sk, VkT = svds(matrix.astype(np.float64), k=self.k)

        # svds returns ascending order -- reverse to descending
        order    = np.argsort(Sk)[::-1]
        self.Uk  = Uk[:, order]       # (vocab_size, k)
        self.Sk  = Sk[order]          # (k,)
        self.Vk  = VkT[order, :].T    # (num_docs, k)

        self.Sk_inv = np.where(self.Sk > 1e-10, 1.0 / self.Sk, 0.0)
        return self

    # -- Query projection -----------------------------------------------------

    def project_query(self, query_tfidf: np.ndarray) -> np.ndarray:
        """
        Projects a TF-IDF query vector into the LSI latent space.

        Formula: q_lsi = query_tfidf @ Uk * Sk^-1

        Args:
            query_tfidf: Dense vector of shape (vocab_size,).

        Returns:
            Dense vector of shape (k,) in latent space.
        """
        return (query_tfidf @ self.Uk) * self.Sk_inv

    # -- Persistence ----------------------------------------------------------

    def save(self) -> None:
        """Saves Uk, Sk, Vk and metadata to data/processed/."""
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        p = _paths(self.k)

        np.save(str(p["Uk"]), self.Uk)
        np.save(str(p["Sk"]), self.Sk)
        np.save(str(p["Vk"]), self.Vk)

        with open(p["meta"], "w") as f:
            json.dump({"k": self.k, "vocab_size": self.Uk.shape[0]}, f)

        print(f"  [SAVED] Uk  -> {p['Uk']}")
        print(f"  [SAVED] Sk  -> {p['Sk']}")
        print(f"  [SAVED] Vk  -> {p['Vk']}")

    @classmethod
    def load(cls, k: int = LSI["k_default"]) -> "LSIModel":
        """
        Loads a previously saved LSI model from disk.

        Args:
            k: Number of latent dimensions of the saved model.

        Returns:
            A fully reconstructed LSIModel.
        """
        p = _paths(k)
        for path in p.values():
            if not path.exists():
                raise FileNotFoundError(
                    f"LSI file not found: {path}. Run the pipeline first."
                )

        model        = cls(k=k)
        model.Uk     = np.load(str(p["Uk"]))
        model.Sk     = np.load(str(p["Sk"]))
        model.Vk     = np.load(str(p["Vk"]))
        model.Sk_inv = np.where(model.Sk > 1e-10, 1.0 / model.Sk, 0.0)
        return model

    # -- Stats ----------------------------------------------------------------

    def explained_variance_ratio(self) -> np.ndarray:
        """Proportion of variance explained by each singular value."""
        sq = self.Sk ** 2
        return sq / sq.sum()

    def cumulative_variance(self) -> np.ndarray:
        """Cumulative explained variance."""
        return np.cumsum(self.explained_variance_ratio())


# -- Entry point --------------------------------------------------------------

def run_lsi(
    tfidf_matrix: sp.csr_matrix,
    k: int = LSI["k_default"],
) -> LSIModel:
    """
    Runs the LSI phase: normalizes, fits SVD and saves model to disk.

    Args:
        tfidf_matrix: Sparse TF-IDF matrix from the indexing phase.
        k:            Number of latent dimensions.

    Returns:
        A fitted LSIModel.
    """
    print("=" * 55)
    print("  LSI-Rank -- LSI model (SVD)")
    print("=" * 55 + "\n")

    model = LSIModel(k=k).fit(tfidf_matrix)

    cum_var = model.cumulative_variance()
    print(f"  [OK] k                     : {k}")
    print(f"  [OK] Variance explained    : {cum_var[-1]:.2%}")
    print(f"  [OK] Top singular value    : {model.Sk[0]:.4f}")
    print(f"  [OK] Bottom singular value : {model.Sk[-1]:.4f}")

    print("\n  Saving model to disk...")
    model.save()

    print("\n" + "=" * 55)
    return model