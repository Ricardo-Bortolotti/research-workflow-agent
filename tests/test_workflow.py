"""Tests for the LangGraph agent workflow."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from agents.concepts_agent import ConceptItem, ConceptResult
from agents.flashcard_agent import FlashcardItem, FlashcardResult
from agents.mindmap_agent import MindMapNode, MindMapResult
from agents.quiz_agent import QuizItem, QuizResult
from agents.summary_agent import SummaryResult
from graph.nodes import WorkflowAgents, create_default_agents
from graph.workflow import WorkflowError, build_workflow, run_workflow


@pytest.fixture
def context_documents() -> list[Document]:
    return [
        Document(
            page_content="The Transformer uses self-attention.",
            metadata={"source": "paper.pdf", "page": 1},
        )
    ]


@pytest.fixture
def mock_agents() -> WorkflowAgents:
    summary_agent = MagicMock()
    summary_agent.run.return_value = SummaryResult(
        executive_summary="Transformer summary.",
        key_insights=["Uses self-attention."],
    )

    concepts_agent = MagicMock()
    concepts_agent.run.return_value = ConceptResult(
        concepts=[ConceptItem("Self-Attention", "Relates sequence positions.", "Core idea.")]
    )

    quiz_agent = MagicMock()
    quiz_agent.run.return_value = QuizResult(
        questions=[QuizItem("What is attention?", "A weighting mechanism.", "easy")]
    )

    flashcard_agent = MagicMock()
    flashcard_agent.run.return_value = FlashcardResult(
        flashcards=[FlashcardItem("Attention", "Mechanism for relating tokens.")]
    )

    mindmap_agent = MagicMock()
    mindmap_agent.run.return_value = MindMapResult(
        title="Transformer",
        nodes=[MindMapNode(label="Self-Attention", children=[])],
    )

    return WorkflowAgents(
        summary=summary_agent,
        concepts=concepts_agent,
        quiz=quiz_agent,
        flashcards=flashcard_agent,
        mindmap=mindmap_agent,
    )


def test_build_workflow_compiles(mock_agents: WorkflowAgents) -> None:
    app = build_workflow(mock_agents)
    assert app is not None


def test_run_workflow_populates_all_outputs(
    mock_agents: WorkflowAgents,
    context_documents: list[Document],
) -> None:
    result = run_workflow("What is the Transformer?", context_documents, agents=mock_agents)

    assert result["summary"]["executive_summary"] == "Transformer summary."
    assert result["concepts"]["concepts"][0]["concept"] == "Self-Attention"
    assert result["quiz"]["questions"][0]["difficulty"] == "easy"
    assert result["flashcards"]["flashcards"][0]["front"] == "Attention"
    assert result["mindmap"]["title"] == "Transformer"
    assert "Self-Attention" in result["mindmap"]["text"]


def test_run_workflow_calls_agents_in_order(
    mock_agents: WorkflowAgents,
    context_documents: list[Document],
) -> None:
    run_workflow("Review Transformer", context_documents, agents=mock_agents)

    assert mock_agents.summary.run.call_count == 1
    assert mock_agents.concepts.run.call_count == 1
    assert mock_agents.quiz.run.call_count == 1
    assert mock_agents.flashcards.run.call_count == 1
    assert mock_agents.mindmap.run.call_count == 1

    mock_agents.summary.run.assert_called_with("Review Transformer", context_documents)
    mock_agents.mindmap.run.assert_called_with("Review Transformer", context_documents)


def test_run_workflow_requires_question(mock_agents: WorkflowAgents, context_documents: list[Document]) -> None:
    with pytest.raises(WorkflowError, match="Question text cannot be empty"):
        run_workflow("  ", context_documents, agents=mock_agents)


def test_run_workflow_requires_context(mock_agents: WorkflowAgents) -> None:
    with pytest.raises(WorkflowError, match="At least one context document"):
        run_workflow("valid question", [], agents=mock_agents)


def test_create_default_agents() -> None:
    agents = create_default_agents(llm=MagicMock())
    assert agents.summary is not None
    assert agents.mindmap is not None
