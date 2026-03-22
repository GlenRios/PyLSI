# PyLSI

Information retrieval system based on **Latent Semantic Indexing (LSI)** over the Cranfield dataset. Uses an inverted index with posting lists for indexing and truncated SVD for semantic retrieval.

## Results (k=150, 225 queries)

| Metric | @5 | @10 | @20 |
|--------|----|-----|-----|
| MAP | **0.3215** | — | — |
| Precision | 0.3262 | 0.2484 | 0.1673 |
| Recall | 0.2961 | 0.4211 | 0.5391 |
| NDCG | 0.3873 | 0.4051 | 0.4454 |

## Project structure

```
lsi-rank/
├── config/
│   └── settings.py          # Global parameters and paths
├── data/
│   ├── raw/                 # Downloaded CRAN files
│   ├── processed/           # documents.jsonl, index.json, tfidf_matrix.npz, lsi_*.npy
│   ├── queries/             # queries.jsonl
│   └── qrels/               # qrels.jsonl
├── src/
│   ├── ingestion/           # Phase 1 — Download and parse
│   │   ├── downloader.py
│   │   └── parser.py
│   ├── preprocessing/       # Phase 2 — Tokenization, stopwords, stemming
│   │   ├── pipeline.py
│   │   ├── stemmer.py
│   │   └── stopwords.py
│   ├── indexing/            # Phase 3 — Posting lists and TF-IDF matrix
│   │   └── posting_lists.py
│   ├── models/              # Phase 4 — Truncated SVD (LSI)
│   │   └── lsi.py
│   ├── retrieval/           # Phase 5 — Query projection and ranking
│   │   └── engine.py
│   └── evaluation/          # Phase 6 — IR metrics
│       └── metrics.py
├── results/
│   ├── metrics/             # eval_k{N}.json per run
│   └── plots/               # comparison.html
├── main.py                  # Pipeline entry point
├── compare_results.py       # Compare multiple k values
└── diagnose.py              # Diagnostic tool
```

## Installation

```bash
pip install numpy scipy scikit-learn
```

No other external dependencies required. The preprocessing pipeline (tokenizer, stemmer, stopwords) is implemented in pure Python.

## Usage

### Run the full pipeline

```bash
python main.py
```

### Set the number of latent dimensions

```bash
python main.py --k 150
```

### Force re-download of the dataset

```bash
python main.py --force --k 150
```

### Compare results across multiple k values

```bash
python main.py --k 50
python main.py --k 100
python main.py --k 150
python main.py --k 200
python main.py --k 300
python compare_results.py
```

Prints a comparison table and generates `results/plots/comparison.html` with interactive charts.

### Diagnose retrieval issues

```bash
python diagnose.py
```

## Dataset — Cranfield

| File | Description | Count |
|------|-------------|-------|
| `cran.all.1400` | Aeronautics academic abstracts | 1,400 docs |
| `cran.qry` | Evaluation queries | 225 queries |
| `cranqrel` | Relevance judgments (scale 1–4, -1) | 1,837 judgments |

Source: [olivernn/cranfield](https://github.com/olivernn/cranfield)

> **Note on query IDs:** The `.I` numbers in `cran.qry` are not sequential. Queries are re-numbered by their position in the file (1–225) to match the sequential IDs used in `cranqrel`.

## Pipeline overview

```
Dataset (CRAN)
    |
    v
Phase 1 — Ingestion       Download + parse -> JSONL
    |
    v
Phase 2 — Preprocessing   Lowercase -> strip -> tokenize -> stopwords -> stem (Porter)
    |
    v
Phase 3 — Indexing        Posting lists -> inverted index -> TF-IDF sparse matrix
    |
    v
Phase 4 — LSI model       Column normalization -> truncated SVD (scipy.sparse.linalg.svds)
    |
    v
Phase 5 — Retrieval       Query projection -> cosine similarity -> top-k ranking
    |
    v
Phase 6 — Evaluation      MAP, Precision@K, Recall@K, NDCG@K
```

## Configuration

All parameters are centralized in `config/settings.py`:

```python
PREPROCESSING = {
    "language":         "english",
    "stemmer":          "porter",
    "remove_stopwords": True,
    "min_token_length": 2,
}

LSI = {
    "k_default": 150,
    "tfidf_sublinear": True,
}

RETRIEVAL = {
    "top_k": 1400,
}

EVALUATION = {
    "cutoffs": [5, 10, 20],
}
```

## Key implementation notes

**TF-IDF formula:**
```
tf_weight  = 1 + log(tf)
idf_weight = log(N / df) + 1
tfidf      = tf_weight * idf_weight
```

**LSI decomposition:**
```
M  ~= Uk . Sk . Vk^T
q_lsi = q_tfidf @ Uk * Sk^-1
```

Column normalization is applied before SVD so that document length does not dominate the decomposition. Relevance judgments use graded scale 1–4; any judgment > 0 is treated as relevant for binary evaluation metrics.