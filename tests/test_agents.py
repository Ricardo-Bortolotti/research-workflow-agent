"""Shared contract tests for all specialized agents."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from agents.concepts_agent import ConceptAgent
from agents.flashcard_agent import FlashcardAgent
from agents.mindmap_agent import MindMapAgent
from agents.quiz_agent import QuizAgent
from agents.summary_agent import SummaryAgent


def _summary_llm() -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = """EXECUTIVE SUMMARY:
A concise summary grounded in the context.

KEY INSIGHTS:
- Insight one
- Insight two
"""
    return llm


def _json_llm(payload: dict[str, Any]) -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = json.dumps(payload)
    return llm


AGENT_CASES = [
    pytest.param(
        SummaryAgent(_summary_llm()),
        id="summary",
    ),
    pytest.param(
        ConceptAgent(
            _json_llm(
                {
                    "concepts": [
                        {
                            "concept": "Attention",
                            "definition": "A weighting mechanism.",
                            "relevance": "Core to the Transformer.",
                        }
                    ]
                }
            )
        ),
        id="concepts",
    ),
    pytest.param(
        QuizAgent(
            _json_llm(
                {
                    "questions": [
                        {
                            "question": "What is attention?",
                            "answer": "A mechanism.",
                            "difficulty": "easy",
                        }
                    ]
                }
            )
        ),
        id="quiz",
    ),
    pytest.param(
        FlashcardAgent(_json_llm({"flashcards": [{"front": "Attention", "back": "Mechanism."}]})),
        id="flashcards",
    ),
    pytest.param(
        MindMapAgent(
            _json_llm(
                {
                    "title": "Transformer",
                    "nodes": [{"label": "Self-Attention", "children": []}],
                }
            )
        ),
        id="mindmap",
    ),
]


@pytest.mark.parametrize("agent", AGENT_CASES)
def test_agents_require_non_empty_question(agent: Any) -> None:
    context = [Document(page_content="context", metadata={"page": 0})]
    with pytest.raises(ValueError, match="Question text cannot be empty"):
        agent.run("   ", context)


@pytest.mark.parametrize("agent", AGENT_CASES)
def test_agents_require_context(agent: Any) -> None:
    with pytest.raises(ValueError, match="At least one context document"):
        agent.run("valid question", [])


@pytest.mark.parametrize("agent", AGENT_CASES)
def test_agents_call_llm_once(agent: Any) -> None:
    context = [
        Document(
            page_content="Transformer architecture details.",
            metadata={"source": "paper.pdf", "page": 1},
        )
    ]
    agent.run("Explain the Transformer", context)
    agent.llm.generate.assert_called_once()
