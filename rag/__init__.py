"""RAG pipeline: document ingestion, chunking, embeddings and retrieval."""

from rag.embeddings import (
    BGE_QUERY_PREFIX,
    DEFAULT_MODEL_NAME,
    ChunkEmbedder,
    EmbeddingError,
)
from rag.loaders import PDFLoadError, load_pdf
from rag.retriever import DEFAULT_TOP_K, DocumentRetriever, RetrieverError
from rag.splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    SplitConfigError,
    create_text_splitter,
    split_documents,
)
from rag.vectorstore import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIR,
    ChromaVectorStore,
    VectorStoreError,
)

__all__ = [
    "BGE_QUERY_PREFIX",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_PERSIST_DIR",
    "DEFAULT_TOP_K",
    "ChunkEmbedder",
    "ChromaVectorStore",
    "DocumentRetriever",
    "EmbeddingError",
    "PDFLoadError",
    "RetrieverError",
    "SplitConfigError",
    "VectorStoreError",
    "create_text_splitter",
    "load_pdf",
    "split_documents",
]
