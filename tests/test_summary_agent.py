"""Tests for the summary agent."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from agents.summary_agent import SummaryAgent, SummaryResult


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = """EXECUTIVE SUMMARY:
The paper introduces the Transformer architecture based on self-attention.

KEY INSIGHTS:
- Self-attention replaces recurrence and convolutions.
- The model scales efficiently on parallel hardware.
- Multi-head attention captures different representation subspaces.
"""
    return llm


@pytest.fixture
def agent(mock_llm: MagicMock) -> SummaryAgent:
    return SummaryAgent(llm=mock_llm)


def test_run_returns_structured_summary(agent: SummaryAgent, mock_llm: MagicMock) -> None:
    context = [
        Document(
            page_content="Attention is all you need.",
            metadata={"source": "paper.pdf", "page": 0},
        )
    ]

    result = agent.run("What is the main contribution?", context)

    assert isinstance(result, SummaryResult)
    assert "Transformer architecture" in result.executive_summary
    assert len(result.key_insights) == 3
    assert "Self-attention replaces recurrence" in result.key_insights[0]
    mock_llm.generate.assert_called_once()
    assert "What is the main contribution?" in mock_llm.generate.call_args.args[0]
    assert "paper.pdf" in mock_llm.generate.call_args.args[0]


def test_run_requires_question(agent: SummaryAgent) -> None:
    context = [Document(page_content="text", metadata={})]
    with pytest.raises(ValueError, match="Question text cannot be empty"):
        agent.run("  ", context)


def test_run_requires_context(agent: SummaryAgent) -> None:
    with pytest.raises(ValueError, match="At least one context document"):
        agent.run("valid question", [])
