"""
ingestion/downloader.py

Downloads the three CRAN dataset files from GitHub:
  - cran.all.1400  ->  1400 academic documents
  - cran.qry       ->  225 evaluation queries
  - cranqrel       ->  relevance judgments
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import CRAN_FILES, RAW_DIR


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """
    Downloads a file from `url` to `dest`.

    Args:
        url:   Source URL.
        dest:  Local destination path.
        force: If True, overwrites even if the file already exists.

    Returns:
        True if downloaded, False if it already existed and force was not set.
    """
    if dest.exists() and not force:
        print(f"  [SKIP] {dest.name} already exists.")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [GET]  {url}")

    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size / 1024
        print(f"  [OK]   {dest.name} ({size_kb:.1f} KB)")
        return True
    except Exception as exc:
        print(f"  [ERR]  Could not download {dest.name}: {exc}")
        raise


def download_cran(force: bool = False) -> None:
    """
    Downloads all CRAN dataset files.

    Args:
        force: If True, re-downloads even if the files already exist.
    """
    print("=" * 55)
    print("  LSI-Rank -- downloading CRAN dataset")
    print("=" * 55)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, info in CRAN_FILES.items():
        print(f"\n[{name.upper()}]")
        download_file(url=info["url"], dest=Path(info["dest"]), force=force)

    print("\n" + "=" * 55)
    print("  Download complete.")
    print(f"  Files saved to: {RAW_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download the CRAN dataset.")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files even if they already exist."
    )
    args = parser.parse_args()
    download_cran(force=args.force)