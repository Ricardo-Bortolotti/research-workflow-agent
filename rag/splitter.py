"""Text splitting utilities for the Book Research Agent RAG pipeline."""

import logging
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


class SplitConfigError(ValueError):
    """Raised when chunking configuration is invalid."""


def _validate_split_config(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise SplitConfigError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise SplitConfigError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise SplitConfigError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
        )


def create_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Create a RecursiveCharacterTextSplitter with the given configuration.

    Args:
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    _validate_split_config(chunk_size, chunk_overlap)

    logger.debug(
        "Creating text splitter (chunk_size=%d, chunk_overlap=%d)",
        chunk_size,
        chunk_overlap,
    )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(
    documents: Sequence[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into smaller chunks for retrieval.

    Args:
        documents: LangChain Document objects (e.g. from load_pdf).
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    if not documents:
        logger.warning("No documents provided for chunking")
        return []

    splitter = create_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    logger.info(
        "Splitting %d document(s) (chunk_size=%d, chunk_overlap=%d)",
        len(documents),
        chunk_size,
        chunk_overlap,
    )

    try:
        chunks = splitter.split_documents(list(documents))
    except Exception as exc:
        logger.exception("Failed to split documents")
        raise SplitConfigError(f"Failed to split documents: {exc}") from exc

    logger.info("Produced %d chunk(s) from %d document(s)", len(chunks), len(documents))
    return chunks
