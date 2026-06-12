"""ChromaDB vector store for the Book Research Agent RAG pipeline."""

import logging
import uuid
from collections.abc import Sequence
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = Path("data/vector_db")
DEFAULT_COLLECTION_NAME = "book_research"


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


def _sanitize_metadata(metadata: dict) -> dict[str, str | int | float | bool]:
    """Convert metadata values to types supported by ChromaDB."""
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized


def _generate_chunk_ids(count: int, prefix: str = "chunk") -> list[str]:
    return [f"{prefix}-{uuid.uuid4().hex}" for _ in range(count)]


def _results_to_documents(results: dict) -> list[Document]:
    documents: list[Document] = []
    doc_list = results.get("documents", [[]])[0]
    meta_list = results.get("metadatas", [[]])[0]
    distance_list = results.get("distances", [[]])[0]

    for index, text in enumerate(doc_list):
        if text is None:
            continue
        metadata = dict(meta_list[index]) if meta_list and meta_list[index] else {}
        if distance_list and distance_list[index] is not None:
            metadata["distance"] = distance_list[index]
        documents.append(Document(page_content=text, metadata=metadata))

    return documents


class ChromaVectorStore:
    """Persistent ChromaDB store for chunk embeddings and semantic retrieval."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._client: chromadb.PersistentClient | None = None
        self._collection: Collection | None = None

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            raise VectorStoreError(
                "Collection not initialized. Call create_collection() or load_collection() first."
            )
        return self._collection

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            logger.info("Opening ChromaDB persistent client at: %s", self.persist_directory.resolve())
            try:
                self._client = chromadb.PersistentClient(path=str(self.persist_directory))
            except Exception as exc:
                logger.exception("Failed to initialize ChromaDB client")
                raise VectorStoreError(f"Failed to initialize ChromaDB client: {exc}") from exc
        return self._client

    def create_collection(self, collection_name: str | None = None, *, reset: bool = False) -> None:
        """Create or open a collection for storing chunk embeddings.

        Args:
            collection_name: Optional collection name override.
            reset: Delete an existing collection before creating a new one.
        """
        name = collection_name or self.collection_name
        client = self._get_client()

        if reset:
            try:
                client.delete_collection(name)
                logger.info("Deleted existing collection: %s", name)
            except Exception:
                logger.debug("No existing collection to delete: %s", name)

        try:
            self._collection = client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.exception("Failed to create collection: %s", name)
            raise VectorStoreError(f"Failed to create collection '{name}': {exc}") from exc

        self.collection_name = name
        logger.info("Collection ready: %s", name)

    def load_collection(self, collection_name: str | None = None) -> None:
        """Load an existing persisted collection."""
        name = collection_name or self.collection_name
        try:
            self._collection = self._get_client().get_collection(name)
        except Exception as exc:
            logger.exception("Failed to load collection: %s", name)
            raise VectorStoreError(f"Failed to load collection '{name}': {exc}") from exc

        self.collection_name = name
        logger.info("Loaded collection: %s", name)

    def add_documents(
        self,
        chunks: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Index chunks and their embeddings in the current collection.

        Args:
            chunks: Chunked documents to store.
            embeddings: Embedding vectors aligned with the chunks.

        Returns:
            Number of indexed chunks.
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return 0

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch"
            )

        ids = _generate_chunk_ids(len(chunks))
        documents = [chunk.page_content for chunk in chunks]
        metadatas = [_sanitize_metadata(chunk.metadata) for chunk in chunks]

        logger.info("Indexing %d chunk(s) into collection '%s'", len(chunks), self.collection_name)

        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=[list(vector) for vector in embeddings],
            )
        except Exception as exc:
            logger.exception("Failed to index chunks")
            raise VectorStoreError(f"Failed to index chunks: {exc}") from exc

        return len(chunks)

    def persist(self) -> None:
        """Confirm local persistence of the vector store.

        ChromaDB's PersistentClient writes data automatically; this method
        validates that the storage directory exists and is writable.
        """
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        logger.info("Vector store persisted at: %s", self.persist_directory.resolve())

    def retrieve(
        self,
        query_embedding: Sequence[float],
        *,
        k: int = 4,
    ) -> list[Document]:
        """Retrieve the most similar chunks for a query embedding.

        Args:
            query_embedding: Dense vector for the search query.
            k: Number of results to return.

        Returns:
            Ranked list of matching Document objects.
        """
        if k <= 0:
            raise VectorStoreError(f"k must be positive, got {k}")

        logger.info("Retrieving top %d chunk(s) from collection '%s'", k, self.collection_name)

        try:
            results = self.collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.exception("Failed to retrieve documents")
            raise VectorStoreError(f"Failed to retrieve documents: {exc}") from exc

        documents = _results_to_documents(results)
        logger.info("Retrieved %d document(s)", len(documents))
        return documents
