"""
ingestion/parser.py

Parses the three CRAN dataset files and saves them as JSONL.

Format of cran.all.1400 and cran.qry:
    .I <id>        -> record identifier
    .T             -> title (documents only)
    .A             -> authors (documents only)
    .B             -> bibliographic source (documents only)
    .W             -> main text (abstract or query body)

Format of cranqrel:
    <query_id> <doc_id> <relevance> <position>
    relevance: 1 (highly relevant) -> 4 (marginally relevant)
               -1 (not relevant)

NOTE on query IDs:
    The .I numbers in cran.qry are NOT sequential 1-225.
    cranqrel uses sequential position numbers (1-225).
    Therefore queries are re-numbered by their position in the file (1-indexed)
    so they match cranqrel exactly.
"""

import json
import re
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    CRAN_FILES,
    PARSED_DOCS_PATH,
    PARSED_QUERIES_PATH,
    PARSED_QRELS_PATH,
    PROCESSED_DIR,
    QUERIES_DIR,
    QRELS_DIR,
)


# -- Utilities ----------------------------------------------------------------

def _read_raw(key: str) -> str:
    """Reads the raw content of one of the CRAN files."""
    path = Path(CRAN_FILES[key]["dest"])
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Run first: python main.py"
        )
    return path.read_text(encoding="utf-8", errors="replace")


def _clean(text: str) -> str:
    """Normalizes whitespace and line breaks in a text block."""
    return re.sub(r"\s+", " ", text).strip()


# -- Parsers ------------------------------------------------------------------

def _iter_records(raw: str, fields: list[str]) -> Iterator[dict]:
    """
    Generic iterator over CRAN records.
    Each record starts with `.I <id>` and contains `.X` sections.

    Args:
        raw:    Full file contents.
        fields: List of fields to extract, e.g. ['T', 'A', 'B', 'W'].

    Yields:
        dict with 'id' (from .I) and the requested fields.
    """
    blocks = re.split(r"\.I\s+", raw.strip())

    for block in blocks:
        if not block.strip():
            continue

        lines  = block.strip().splitlines()
        doc_id = int(lines[0].strip())
        record: dict = {"id": doc_id}

        for field in fields:
            pattern = rf"\.{field}\s*(.*?)(?=\.[A-Z]\b|$)"
            match   = re.search(pattern, "\n".join(lines[1:]), re.DOTALL)
            record[field.lower()] = _clean(match.group(1)) if match else ""

        yield record


def parse_documents() -> list[dict]:
    """
    Parses cran.all.1400.
    Document IDs come from .I and are 1-1400 (sequential — no remapping needed).

    Returns:
        List of dicts with keys: id, t (title), a (authors), b (source), w (abstract).
    """
    raw  = _read_raw("documents")
    docs = list(_iter_records(raw, fields=["T", "A", "B", "W"]))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARSED_DOCS_PATH, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"  [DOCS]    {len(docs)} documents -> {PARSED_DOCS_PATH}")
    return docs


def parse_queries() -> list[dict]:
    """
    Parses cran.qry.

    IMPORTANT: The .I numbers in cran.qry are NOT sequential 1-225 and
    do NOT match cranqrel. Queries are re-numbered by their 1-based position
    in the file so they align with the sequential IDs used in cranqrel.

    Returns:
        List of dicts with keys: id (sequential 1-225), raw_id (.I number), w (text).
    """
    raw     = _read_raw("queries")
    records = list(_iter_records(raw, fields=["W"]))

    # Re-number sequentially: position 1, 2, ..., 225
    queries = []
    for seq_id, record in enumerate(records, start=1):
        queries.append({
            "id":     seq_id,           # sequential ID matching cranqrel
            "raw_id": record["id"],     # original .I number (kept for reference)
            "w":      record["w"],
        })

    QUERIES_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARSED_QUERIES_PATH, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"  [QUERIES] {len(queries)} queries -> {PARSED_QUERIES_PATH}")
    return queries


def parse_qrels() -> list[dict]:
    """
    Parses cranqrel (relevance judgments).

    Line format: query_id  doc_id  relevance  position
    query_id is sequential 1-225, matching the re-numbered query IDs above.

    Returns:
        List of dicts with keys: query_id, doc_id, relevance, relevant.
        relevance 1-4 -> positive (relevant=True).
        relevance -1  -> not relevant (relevant=False).
    """
    raw   = _read_raw("qrels")
    qrels = []

    for line in raw.strip().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        query_id  = int(parts[0])
        doc_id    = int(parts[1])
        relevance = int(parts[2])

        qrels.append({
            "query_id":  query_id,
            "doc_id":    doc_id,
            "relevance": relevance,
            "relevant":  relevance > 0,
        })

    QRELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARSED_QRELS_PATH, "w", encoding="utf-8") as f:
        for qr in qrels:
            f.write(json.dumps(qr, ensure_ascii=False) + "\n")

    print(f"  [QRELS]   {len(qrels)} relevance judgments -> {PARSED_QRELS_PATH}")
    return qrels


# -- Loaders ------------------------------------------------------------------

def load_documents() -> list[dict]:
    """Loads parsed documents from JSONL."""
    return [json.loads(l) for l in PARSED_DOCS_PATH.read_text().splitlines()]


def load_queries() -> list[dict]:
    """Loads parsed queries from JSONL."""
    return [json.loads(l) for l in PARSED_QUERIES_PATH.read_text().splitlines()]


def load_qrels() -> list[dict]:
    """Loads relevance judgments from JSONL."""
    return [json.loads(l) for l in PARSED_QRELS_PATH.read_text().splitlines()]


def load_qrels_dict() -> dict[int, dict[int, bool]]:
    """
    Converts qrels to a nested dict for O(1) lookup.

    Returns:
        {query_id: {doc_id: relevant (bool)}}
    """
    result: dict[int, dict[int, bool]] = {}
    for qr in load_qrels():
        qid = qr["query_id"]
        result.setdefault(qid, {})[qr["doc_id"]] = qr["relevant"]
    return result


# -- CLI ----------------------------------------------------------------------

def parse_all() -> None:
    """Parses and persists all three CRAN files."""
    print("=" * 55)
    print("  LSI-Rank -- parsing CRAN dataset")
    print("=" * 55 + "\n")

    docs    = parse_documents()
    queries = parse_queries()
    qrels   = parse_qrels()

    print("\n-- Summary ------------------------------------------")
    print(f"  Documents    : {len(docs)}")
    print(f"  Queries      : {len(queries)}")
    print(f"  Qrels        : {len(qrels)}")

    relevant_count = sum(1 for qr in qrels if qr["relevant"])
    print(f"  Relevant     : {relevant_count} / {len(qrels)}")
    print("=" * 55)


if __name__ == "__main__":
    parse_all()