"""Document loaders for the Book Research Agent RAG pipeline."""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf"}


class PDFLoadError(Exception):
    """Raised when a PDF document cannot be loaded."""


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a PDF file and return one LangChain Document per page.

    Args:
        file_path: Path to a local PDF file.

    Returns:
        List of Document objects with page content and metadata (source, page).

    Raises:
        PDFLoadError: If the path is invalid, the file is not a PDF, or parsing fails.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        logger.error("PDF file not found: %s", path)
        raise PDFLoadError(f"File not found: {path}")

    if not path.is_file():
        logger.error("Path is not a file: %s", path)
        raise PDFLoadError(f"Path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.error("Unsupported file extension '%s' for: %s", path.suffix, path)
        raise PDFLoadError(
            f"Unsupported file extension '{path.suffix}'. Expected one of: {SUPPORTED_EXTENSIONS}"
        )

    logger.info("Loading PDF: %s", path)

    try:
        documents = PyPDFLoader(str(path)).load()
    except Exception as exc:
        logger.exception("Failed to parse PDF: %s", path)
        raise PDFLoadError(f"Failed to parse PDF '{path.name}': {exc}") from exc

    if not documents:
        logger.warning("PDF contains no extractable pages: %s", path)
        raise PDFLoadError(f"PDF contains no extractable pages: {path.name}")

    logger.info("Loaded %d page(s) from PDF: %s", len(documents), path.name)
    return documents
