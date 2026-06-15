"""Embedding generation via the Hugging Face Inference API."""

import logging
import os
from collections.abc import Sequence

import numpy as np
from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError, HfHubHTTPError
from langchain_core.documents import Document

from app.hf_auth import resolve_hf_api_token

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_ENV_VAR = "HF_EMBEDDING_MODEL_ID"
BGE_SMALL_DIMENSION = 384
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


class ChunkEmbedder:
    """Generate dense semantic vectors using the Hugging Face Inference API."""

    def __init__(
        self,
        model_name: str | None = None,
        api_token: str | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv(EMBEDDING_MODEL_ENV_VAR, DEFAULT_MODEL_NAME)
        try:
            self.api_token = api_token or resolve_hf_api_token()
        except ValueError as exc:
            raise EmbeddingError(str(exc)) from exc
        self._client = InferenceClient(token=self.api_token)

    @property
    def embedding_dimension(self) -> int:
        """Return the output dimensionality of the embedding model."""
        return BGE_SMALL_DIMENSION

    def embed_texts(self, texts: Sequence[str], *, normalize: bool = True) -> list[list[float]]:
        """Embed a sequence of raw text strings."""
        if not texts:
            logger.warning("No texts provided for embedding")
            return []

        logger.info("Embedding %d text(s) with model %s via HF API", len(texts), self.model_name)

        vectors: list[list[float]] = []
        try:
            for text in texts:
                raw_vector = self._client.feature_extraction(text, model=self.model_name)
                vectors.append(_coerce_vector(raw_vector))
        except (BadRequestError, HfHubHTTPError) as exc:
            logger.exception("Hugging Face embedding API request failed")
            raise EmbeddingError(f"Hugging Face embedding API failed: {exc}") from exc
        except Exception as exc:
            logger.exception("Failed to generate embeddings")
            raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc

        if normalize:
            vectors = [_l2_normalize(vector) for vector in vectors]

        return vectors

    def embed_chunks(
        self, chunks: Sequence[Document], *, normalize: bool = True
    ) -> list[list[float]]:
        """Embed LangChain Document chunks using their page_content."""
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []

        texts = [chunk.page_content for chunk in chunks]
        return self.embed_texts(texts, normalize=normalize)

    def embed_query(self, query: str, *, normalize: bool = True) -> list[float]:
        """Embed a search query using the BGE retrieval instruction prefix."""
        if not query.strip():
            raise EmbeddingError("Query text cannot be empty")

        prefixed_query = f"{BGE_QUERY_PREFIX}{query}"
        vectors = self.embed_texts([prefixed_query], normalize=normalize)
        return vectors[0]


def _coerce_vector(raw_vector: object) -> list[float]:
    """Normalize HF API responses into a flat float vector."""
    array = np.asarray(raw_vector, dtype=float)
    if array.ndim > 1:
        array = array.reshape(-1)
    return array.tolist()


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    if norm == 0:
        return array.tolist()
    return (array / norm).tolist()
