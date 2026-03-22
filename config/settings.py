"""
Central project configuration for LSI-Rank.
All global parameters and paths are defined here.
"""

from pathlib import Path

# -- Base paths ---------------------------------------------------------------
ROOT_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
QUERIES_DIR    = DATA_DIR / "queries"
QRELS_DIR      = DATA_DIR / "qrels"
RESULTS_DIR    = ROOT_DIR / "results"
METRICS_DIR    = RESULTS_DIR / "metrics"
PLOTS_DIR      = RESULTS_DIR / "plots"

# -- CRAN dataset URLs --------------------------------------------------------
CRAN_BASE_URL = "https://raw.githubusercontent.com/olivernn/cranfield/master/data/cran"

CRAN_FILES = {
    "documents": {
        "url":  f"{CRAN_BASE_URL}/cran.all.1400",
        "dest": RAW_DIR / "cran.all.1400",
    },
    "queries": {
        "url":  f"{CRAN_BASE_URL}/cran.qry",
        "dest": RAW_DIR / "cran.qry",
    },
    "qrels": {
        "url":  f"{CRAN_BASE_URL}/cranqrel",
        "dest": RAW_DIR / "cranqrel",
    },
}

# -- Parsed output paths ------------------------------------------------------
PARSED_DOCS_PATH    = PROCESSED_DIR / "documents.jsonl"
PARSED_QUERIES_PATH = QUERIES_DIR   / "queries.jsonl"
PARSED_QRELS_PATH   = QRELS_DIR     / "qrels.jsonl"

# -- Preprocessing ------------------------------------------------------------
PREPROCESSING = {
    "language":         "english",
    "stemmer":          "porter",       # "porter" | "snowball" | None
    "remove_stopwords": True,
    "min_token_length": 2,
}

# -- LSI model ----------------------------------------------------------------
LSI = {
    "k_values":        [50, 100, 150, 200, 300],  # dimensions to experiment with
    "k_default":       150,
    "tfidf_sublinear": True,
}

# -- Retrieval ----------------------------------------------------------------
RETRIEVAL = {
    "top_k": 1400,
}

# -- Evaluation ---------------------------------------------------------------
EVALUATION = {
    "cutoffs": [5, 10, 20],   # Precision@K and Recall@K cutoffs
}