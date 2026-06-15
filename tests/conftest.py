"""Shared pytest fixtures for the Book Research Agent test suite."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_context() -> list[Document]:
    """Minimal RAG context used across agent tests."""
    return [
        Document(
            page_content="The Transformer uses self-attention and multi-head attention.",
            metadata={"source": "paper.pdf", "page": 2},
        )
    ]


@pytest.fixture
def mock_llm() -> MagicMock:
    """Generic mocked LLM for agents that do not need structured output."""
    llm = MagicMock()
    llm.generate.return_value = "mocked llm response"
    return llm
