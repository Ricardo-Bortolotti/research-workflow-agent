"""Tests for the concept agent."""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from agents.concepts_agent import (
    ConceptAgent,
    ConceptAgentError,
    ConceptItem,
    ConceptResult,
)


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = json.dumps(
        {
            "concepts": [
                {
                    "concept": "Self-Attention",
                    "definition": "A mechanism that relates positions in a sequence.",
                    "relevance": "Core building block of the Transformer.",
                },
                {
                    "concept": "Multi-Head Attention",
                    "definition": "Parallel attention heads over representation subspaces.",
                    "relevance": "Improves the model's ability to capture diverse patterns.",
                },
            ]
        }
    )
    return llm


@pytest.fixture
def agent(mock_llm: MagicMock) -> ConceptAgent:
    return ConceptAgent(llm=mock_llm)


def test_run_returns_structured_concepts(agent: ConceptAgent, mock_llm: MagicMock) -> None:
    context = [
        Document(
            page_content="The Transformer uses self-attention and multi-head attention.",
            metadata={"source": "paper.pdf", "page": 2},
        )
    ]

    result = agent.run("What are the key ideas?", context)

    assert isinstance(result, ConceptResult)
    assert len(result.concepts) == 2
    assert result.concepts[0].concept == "Self-Attention"
    assert "sequence" in result.concepts[0].definition
    assert "Transformer" in result.concepts[0].relevance
    mock_llm.generate.assert_called_once()


def test_to_json_serializes_concepts(agent: ConceptAgent) -> None:
    context = [Document(page_content="Attention mechanisms.", metadata={"page": 1})]
    result = agent.run("Explain attention", context)

    payload = json.loads(result.to_json())
    assert payload["concepts"][0]["concept"] == "Self-Attention"
    assert set(payload["concepts"][0]) == {"concept", "definition", "relevance"}


def test_run_requires_question(agent: ConceptAgent) -> None:
    context = [Document(page_content="text", metadata={})]
    with pytest.raises(ValueError, match="Question text cannot be empty"):
        agent.run("  ", context)


def test_run_requires_context(agent: ConceptAgent) -> None:
    with pytest.raises(ValueError, match="At least one context document"):
        agent.run("valid question", [])


def test_invalid_json_raises(agent: ConceptAgent, mock_llm: MagicMock) -> None:
    mock_llm.generate.return_value = "not json"
    context = [Document(page_content="text", metadata={})]

    with pytest.raises(ConceptAgentError, match="Invalid JSON"):
        agent.run("question", context)


def test_missing_fields_raise(agent: ConceptAgent, mock_llm: MagicMock) -> None:
    mock_llm.generate.return_value = json.dumps({"concepts": [{"concept": "Only name"}]})
    context = [Document(page_content="text", metadata={})]

    with pytest.raises(ConceptAgentError, match="missing required fields"):
        agent.run("question", context)


def test_concept_item_dataclass() -> None:
    item = ConceptItem(
        concept="Transformer",
        definition="An attention-based architecture.",
        relevance="Main model proposed in the paper.",
    )
    assert item.concept == "Transformer"
