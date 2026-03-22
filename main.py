
"""
main.py — LSI-Rank

Project entry point. Runs all phases in order.

Usage:
    python main.py           # downloads files if they don't exist
    python main.py --force   # always re-downloads
    python main.py --k 200   # set LSI dimensions
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingestion.downloader import download_cran
from src.ingestion.parser import parse_all, load_documents, load_queries, load_qrels_dict
from src.preprocessing.pipeline import run_preprocessing
from src.indexing.posting_lists import run_indexing
from src.models.lsi import run_lsi
from src.retrieval.engine import run_retrieval_demo
from src.evaluation.metrics import run_evaluation
from config.settings import LSI


def main() -> None:
    parser = argparse.ArgumentParser(description="LSI-Rank pipeline.")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files even if they already exist."
    )
    parser.add_argument(
        "--k", type=int, default=LSI["k_default"],
        help=f"Number of latent dimensions for LSI (default: {LSI['k_default']})."
    )
    args = parser.parse_args()

    # -- Phase 1: Download and parse ------------------------------------------
    download_cran(force=args.force)
    print()
    parse_all()

    # -- Phase 2: Preprocessing -----------------------------------------------
    docs, queries = load_documents(), load_queries()
    processed_docs, processed_queries = run_preprocessing(docs, queries)

    # -- Phase 3: Indexing ----------------------------------------------------
    index, tfidf_matrix = run_indexing(processed_docs)

    # -- Phase 4: LSI model ---------------------------------------------------
    lsi_model = run_lsi(tfidf_matrix, k=args.k)

    # -- Phase 5: Retrieval ---------------------------------------------------
    run_retrieval_demo(processed_queries, index, lsi_model)

    # -- Phase 6: Evaluation --------------------------------------------------
    qrels_dict = load_qrels_dict()
    run_evaluation(processed_queries, qrels_dict, index, lsi_model)


if __name__ == "__main__":
    main()