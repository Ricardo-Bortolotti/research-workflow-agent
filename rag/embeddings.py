"""Embedding generation for the Book Research Agent RAG pipeline."""

import logging
from collections.abc import Sequence

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


class ChunkEmbedder:
    """Generates dense semantic vectors for document chunks using sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def embedding_dimension(self) -> int:
        """Return the output dimensionality of the embedding model."""
        model = self._get_model()
        if hasattr(model, "get_embedding_dimension"):
            return model.get_embedding_dimension()
        return model.get_sentence_embedding_dimension()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                logger.exception("Failed to load embedding model: %s", self.model_name)
                raise EmbeddingError(f"Failed to load model '{self.model_name}': {exc}") from exc
        return self._model

    def embed_texts(self, texts: Sequence[str], *, normalize: bool = True) -> list[list[float]]:
        """Embed a sequence of raw text strings.

        Args:
            texts: Text passages to embed (e.g. chunk contents).
            normalize: L2-normalize vectors for cosine-similarity search.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            logger.warning("No texts provided for embedding")
            return []

        logger.info("Embedding %d text(s) with model %s", len(texts), self.model_name)

        try:
            vectors = self._get_model().encode(
                list(texts),
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.exception("Failed to generate embeddings")
            raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc

        return np.asarray(vectors).tolist()

    def embed_chunks(self, chunks: Sequence[Document], *, normalize: bool = True) -> list[list[float]]:
        """Embed LangChain Document chunks using their page_content.

        Args:
            chunks: Chunked documents from the splitting step.
            normalize: L2-normalize vectors for cosine-similarity search.

        Returns:
            List of embedding vectors aligned with the input chunk order.
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []

        texts = [chunk.page_content for chunk in chunks]
        return self.embed_texts(texts, normalize=normalize)

    def embed_query(self, query: str, *, normalize: bool = True) -> list[float]:
        """Embed a search query using the BGE retrieval instruction prefix.

        Args:
            query: Natural-language question or search phrase.
            normalize: L2-normalize the vector for cosine-similarity search.

        Returns:
            Single embedding vector for the query.
        """
        if not query.strip():
            raise EmbeddingError("Query text cannot be empty")

        prefixed_query = f"{BGE_QUERY_PREFIX}{query}"
        vectors = self.embed_texts([prefixed_query], normalize=normalize)
        return vectors[0]
