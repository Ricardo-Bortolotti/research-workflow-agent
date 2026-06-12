"""RAG retriever: similarity search over indexed chunks in ChromaDB.

Examples
--------
Load an existing index and retrieve relevant chunks:

    from rag.retriever import DocumentRetriever
    from rag.vectorstore import ChromaVectorStore

    store = ChromaVectorStore()
    store.load_collection()

    retriever = DocumentRetriever(vector_store=store, top_k=4)
    chunks = retriever.invoke("What is the attention mechanism?")

    for chunk in chunks:
        print(chunk.page_content[:200])
        print(chunk.metadata)

Override ``top_k`` for a single query:

    chunks = retriever.invoke("Explain the transformer architecture", top_k=6)

Full indexing + retrieval pipeline:

    from rag.embeddings import ChunkEmbedder
    from rag.loaders import load_pdf
    from rag.retriever import DocumentRetriever
    from rag.splitter import split_documents
    from rag.vectorstore import ChromaVectorStore

    documents = load_pdf("data/raw/sample.pdf")
    chunks = split_documents(documents)

    embedder = ChunkEmbedder()
    vectors = embedder.embed_chunks(chunks)

    store = ChromaVectorStore()
    store.create_collection(reset=True)
    store.add_documents(chunks, vectors)
    store.persist()

    retriever = DocumentRetriever(vector_store=store, embedder=embedder, top_k=3)
    results = retriever.invoke("How does self-attention work?")
"""

import logging

from langchain_core.documents import Document

from rag.embeddings import ChunkEmbedder
from rag.vectorstore import ChromaVectorStore

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 4


class RetrieverError(Exception):
    """Raised when retrieval fails."""


class DocumentRetriever:
    """Retrieve the most relevant document chunks for a natural-language query."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: ChunkEmbedder | None = None,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder or ChunkEmbedder()
        self.top_k = top_k
        self._validate_top_k(top_k)

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if top_k <= 0:
            raise RetrieverError(f"top_k must be positive, got {top_k}")

    def invoke(self, query: str, *, top_k: int | None = None) -> list[Document]:
        """Run similarity search and return the top-k most relevant chunks.

        Args:
            query: Natural-language question or search phrase.
            top_k: Optional per-query override for the number of results.

        Returns:
            Ranked list of relevant Document chunks.
        """
        if not query.strip():
            raise RetrieverError("Query text cannot be empty")

        k = top_k if top_k is not None else self.top_k
        self._validate_top_k(k)

        logger.info("Retrieving top %d chunk(s) for query: %r", k, query)

        try:
            query_embedding = self.embedder.embed_query(query)
            documents = self.vector_store.retrieve(query_embedding, k=k)
        except RetrieverError:
            raise
        except Exception as exc:
            logger.exception("Retrieval failed for query: %r", query)
            raise RetrieverError(f"Retrieval failed: {exc}") from exc

        logger.info("Returning %d relevant chunk(s)", len(documents))
        return documents

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[Document]:
        """Alias for :meth:`invoke`."""
        return self.invoke(query, top_k=top_k)
