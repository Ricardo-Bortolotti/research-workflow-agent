"""Tests for the flashcard agent."""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from agents.flashcard_agent import (
    FlashcardAgent,
    FlashcardAgentError,
    FlashcardItem,
    FlashcardResult,
)


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = json.dumps(
        {
            "flashcards": [
                {
                    "front": "What is self-attention?",
                    "back": "A mechanism relating sequence positions.",
                },
                {"front": "Multi-head attention", "back": "Parallel attention over subspaces."},
            ]
        }
    )
    return llm


@pytest.fixture
def agent(mock_llm: MagicMock) -> FlashcardAgent:
    return FlashcardAgent(llm=mock_llm)


def test_run_returns_structured_flashcards(agent: FlashcardAgent) -> None:
    context = [Document(page_content="Transformer uses self-attention.", metadata={"page": 1})]
    result = agent.run("Study Transformer concepts", context)

    assert isinstance(result, FlashcardResult)
    assert len(result.flashcards) == 2
    assert result.flashcards[0].front == "What is self-attention?"
    assert "mechanism" in result.flashcards[0].back


def test_to_json_serializes_flashcards(agent: FlashcardAgent) -> None:
    context = [Document(page_content="Attention.", metadata={})]
    result = agent.run("Flashcards", context)

    payload = json.loads(result.to_json())
    assert set(payload["flashcards"][0]) == {"front", "back"}


def test_run_requires_question(agent: FlashcardAgent) -> None:
    context = [Document(page_content="text", metadata={})]
    with pytest.raises(ValueError, match="Question text cannot be empty"):
        agent.run("  ", context)


def test_run_requires_context(agent: FlashcardAgent) -> None:
    with pytest.raises(ValueError, match="At least one context document"):
        agent.run("valid question", [])


def test_invalid_json_raises(agent: FlashcardAgent, mock_llm: MagicMock) -> None:
    mock_llm.generate.return_value = "not json"
    context = [Document(page_content="text", metadata={})]

    with pytest.raises(FlashcardAgentError, match="Invalid JSON"):
        agent.run("question", context)


def test_missing_fields_raise(agent: FlashcardAgent, mock_llm: MagicMock) -> None:
    mock_llm.generate.return_value = json.dumps({"flashcards": [{"front": "Only front"}]})
    context = [Document(page_content="text", metadata={})]

    with pytest.raises(FlashcardAgentError, match="missing required fields"):
        agent.run("question", context)


def test_flashcard_item_dataclass() -> None:
    item = FlashcardItem(front="Term", back="Definition")
    assert item.front == "Term"
