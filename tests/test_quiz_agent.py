"""Tests for the quiz agent."""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from agents.quiz_agent import QuizAgent, QuizAgentError, QuizItem, QuizResult


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = json.dumps(
        {
            "questions": [
                {
                    "question": "What mechanism does the Transformer rely on?",
                    "answer": "Self-attention.",
                    "difficulty": "easy",
                },
                {
                    "question": "Why is multi-head attention used?",
                    "answer": "To attend to information from different representation subspaces.",
                    "difficulty": "medium",
                },
                {
                    "question": "How does self-attention reduce computational path length?",
                    "answer": "It connects all positions with a constant number of operations.",
                    "difficulty": "hard",
                },
            ]
        }
    )
    return llm


@pytest.fixture
def agent(mock_llm: MagicMock) -> QuizAgent:
    return QuizAgent(llm=mock_llm)


def test_run_returns_structured_quiz(agent: QuizAgent, mock_llm: MagicMock) -> None:
    context = [
        Document(
            page_content="The Transformer uses self-attention and multi-head attention.",
            metadata={"source": "paper.pdf", "page": 2},
        )
    ]

    result = agent.run("Review Transformer concepts", context)

    assert isinstance(result, QuizResult)
    assert len(result.questions) == 3
    assert result.questions[0].difficulty == "easy"
    assert "Self-attention" in result.questions[0].answer
    mock_llm.generate.assert_called_once()


def test_to_json_serializes_questions(agent: QuizAgent) -> None:
    context = [Document(page_content="Attention mechanisms.", metadata={"page": 1})]
    result = agent.run("Quiz me on attention", context)

    payload = json.loads(result.to_json())
    assert payload["questions"][0]["question"].startswith("What mechanism")
    assert set(payload["questions"][0]) == {"question", "answer", "difficulty"}


def test_run_requires_question(agent: QuizAgent) -> None:
    context = [Document(page_content="text", metadata={})]
    with pytest.raises(ValueError, match="Question text cannot be empty"):
        agent.run("  ", context)


def test_run_requires_context(agent: QuizAgent) -> None:
    with pytest.raises(ValueError, match="At least one context document"):
        agent.run("valid question", [])


def test_invalid_json_raises(agent: QuizAgent, mock_llm: MagicMock) -> None:
    mock_llm.generate.return_value = "not json"
    context = [Document(page_content="text", metadata={})]

    with pytest.raises(QuizAgentError, match="Invalid JSON"):
        agent.run("question", context)


def test_missing_fields_raise(agent: QuizAgent, mock_llm: MagicMock) -> None:
    mock_llm.generate.return_value = json.dumps({"questions": [{"question": "Only question?"}]})
    context = [Document(page_content="text", metadata={})]

    with pytest.raises(QuizAgentError, match="missing required fields"):
        agent.run("question", context)


def test_invalid_difficulty_raises(agent: QuizAgent, mock_llm: MagicMock) -> None:
    mock_llm.generate.return_value = json.dumps(
        {
            "questions": [
                {
                    "question": "What is attention?",
                    "answer": "A weighting mechanism.",
                    "difficulty": "expert",
                }
            ]
        }
    )
    context = [Document(page_content="text", metadata={})]

    with pytest.raises(QuizAgentError, match="invalid difficulty"):
        agent.run("question", context)


def test_quiz_item_dataclass() -> None:
    item = QuizItem(
        question="What is a Transformer?",
        answer="An attention-based architecture.",
        difficulty="medium",
    )
    assert item.difficulty == "medium"
