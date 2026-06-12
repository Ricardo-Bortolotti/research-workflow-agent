"""LangGraph workflow orchestrating all Book Research Agent nodes."""

import logging
from collections.abc import Sequence

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.llm import HuggingFaceLLM
from graph.nodes import (
    WorkflowAgents,
    create_concepts_node,
    create_default_agents,
    create_flashcards_node,
    create_mindmap_node,
    create_quiz_node,
    create_summary_node,
)
from graph.state import WorkflowState

logger = logging.getLogger(__name__)

SUMMARY_NODE = "summary"
CONCEPTS_NODE = "concepts"
QUIZ_NODE = "quiz"
FLASHCARDS_NODE = "flashcards"
MINDMAP_NODE = "mindmap"


class WorkflowError(Exception):
    """Raised when workflow input validation fails."""


def build_workflow(agents: WorkflowAgents | None = None) -> CompiledStateGraph:
    """Build and compile the linear agent workflow DAG.

    Flow:
        SummaryAgent → ConceptAgent → QuizAgent → FlashcardAgent → MindMapAgent

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    workflow_agents = agents or create_default_agents()

    graph = StateGraph(WorkflowState)

    graph.add_node(SUMMARY_NODE, create_summary_node(workflow_agents.summary))
    graph.add_node(CONCEPTS_NODE, create_concepts_node(workflow_agents.concepts))
    graph.add_node(QUIZ_NODE, create_quiz_node(workflow_agents.quiz))
    graph.add_node(FLASHCARDS_NODE, create_flashcards_node(workflow_agents.flashcards))
    graph.add_node(MINDMAP_NODE, create_mindmap_node(workflow_agents.mindmap))

    graph.add_edge(START, SUMMARY_NODE)
    graph.add_edge(SUMMARY_NODE, CONCEPTS_NODE)
    graph.add_edge(CONCEPTS_NODE, QUIZ_NODE)
    graph.add_edge(QUIZ_NODE, FLASHCARDS_NODE)
    graph.add_edge(FLASHCARDS_NODE, MINDMAP_NODE)
    graph.add_edge(MINDMAP_NODE, END)

    logger.info("Compiled workflow DAG: summary → concepts → quiz → flashcards → mindmap")
    return graph.compile()


def run_workflow(
    question: str,
    context_documents: Sequence[Document],
    *,
    agents: WorkflowAgents | None = None,
    llm: HuggingFaceLLM | None = None,
) -> WorkflowState:
    """Execute the full agent workflow on retrieved context.

    Args:
        question: Research question guiding all agents.
        context_documents: Chunks retrieved by the RAG retriever.
        agents: Optional preconfigured agent instances for testing or customization.
        llm: Optional shared LLM client when agents are not provided.

    Returns:
        Final workflow state with all agent outputs populated.
    """
    if not question.strip():
        raise WorkflowError("Question text cannot be empty")
    if not context_documents:
        raise WorkflowError("At least one context document is required")

    workflow_agents = agents or create_default_agents(llm)
    app = build_workflow(workflow_agents)

    initial_state: WorkflowState = {
        "question": question.strip(),
        "context_documents": list(context_documents),
    }

    logger.info("Running workflow for question: %r", question)
    return app.invoke(initial_state)
