"""LangGraph node functions for the Book Research Agent workflow."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from agents.concepts_agent import ConceptAgent
from agents.flashcard_agent import FlashcardAgent
from agents.mindmap_agent import MindMapAgent
from agents.quiz_agent import QuizAgent
from agents.summary_agent import SummaryAgent, SummaryResult
from app.llm import HuggingFaceLLM
from graph.state import WorkflowState

logger = logging.getLogger(__name__)

NodeFn = Callable[[WorkflowState], dict]


@dataclass
class WorkflowAgents:
    """Container for all agents used in the workflow."""

    summary: SummaryAgent
    concepts: ConceptAgent
    quiz: QuizAgent
    flashcards: FlashcardAgent
    mindmap: MindMapAgent


def create_default_agents(llm: HuggingFaceLLM | None = None) -> WorkflowAgents:
    """Instantiate all workflow agents, optionally sharing one LLM client."""
    shared_llm = llm or HuggingFaceLLM()
    return WorkflowAgents(
        summary=SummaryAgent(shared_llm),
        concepts=ConceptAgent(shared_llm),
        quiz=QuizAgent(shared_llm),
        flashcards=FlashcardAgent(shared_llm),
        mindmap=MindMapAgent(shared_llm),
    )


def create_summary_node(agent: SummaryAgent) -> NodeFn:
    """Create the SummaryAgent node."""

    def summary_node(state: WorkflowState) -> dict:
        logger.info("Workflow node: summary")
        result = agent.run(state["question"], state["context_documents"])
        return {"summary": _summary_to_dict(result)}

    return summary_node


def create_concepts_node(agent: ConceptAgent) -> NodeFn:
    """Create the ConceptAgent node."""

    def concepts_node(state: WorkflowState) -> dict:
        logger.info("Workflow node: concepts")
        result = agent.run(state["question"], state["context_documents"])
        return {"concepts": result.to_dict()}

    return concepts_node


def create_quiz_node(agent: QuizAgent) -> NodeFn:
    """Create the QuizAgent node."""

    def quiz_node(state: WorkflowState) -> dict:
        logger.info("Workflow node: quiz")
        result = agent.run(state["question"], state["context_documents"])
        return {"quiz": result.to_dict()}

    return quiz_node


def create_flashcards_node(agent: FlashcardAgent) -> NodeFn:
    """Create the FlashcardAgent node."""

    def flashcards_node(state: WorkflowState) -> dict:
        logger.info("Workflow node: flashcards")
        result = agent.run(state["question"], state["context_documents"])
        return {"flashcards": result.to_dict()}

    return flashcards_node


def create_mindmap_node(agent: MindMapAgent) -> NodeFn:
    """Create the MindMapAgent node."""

    def mindmap_node(state: WorkflowState) -> dict:
        logger.info("Workflow node: mindmap")
        result = agent.run(state["question"], state["context_documents"])
        return {"mindmap": {**result.to_dict(), "text": result.to_text()}}

    return mindmap_node


def _summary_to_dict(result: SummaryResult) -> dict:
    return {
        "executive_summary": result.executive_summary,
        "key_insights": result.key_insights,
    }
