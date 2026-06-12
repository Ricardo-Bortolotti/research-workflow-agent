"""Manual script to test PDF ingestion."""

import argparse
import sys
from pathlib import Path

from rag.loaders import load_pdf

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = ROOT / "data" / "raw" / "sample.pdf"


def resolve_pdf_path(filename: str) -> Path:
    """Resolve filename to a path inside data/raw/ or use the given path as-is."""
    path = Path(filename)
    if path.is_absolute() or path.parent != Path("."):
        return path.resolve()
    return (ROOT / "data" / "raw" / path.name).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test PDF ingestion with the project loader.")
    parser.add_argument(
        "pdf",
        nargs="?",
        default=str(DEFAULT_PDF),
        help="PDF filename in data/raw/ (e.g. book.pdf) or full path",
    )
    args = parser.parse_args()

    pdf_path = resolve_pdf_path(args.pdf)
    docs = load_pdf(pdf_path)

    print(f"File: {pdf_path}")
    print(f"Documents loaded: {len(docs)}")

    print("\nFirst 500 characters:\n")
    print(docs[0].page_content[:500])

    print("\nMetadata:\n")
    print(docs[0].metadata)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
