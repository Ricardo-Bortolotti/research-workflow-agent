"""Shared state definition for the Book Research Agent LangGraph workflow."""

from typing import Any, TypedDict

from langchain_core.documents import Document


class WorkflowState(TypedDict, total=False):
    """State passed between nodes in the agent workflow DAG.

    Required inputs:
        question: Research question guiding all agents.
        context_documents: Retrieved chunks from the RAG retriever.

    Populated outputs (one per agent node):
        summary: Executive summary and key insights.
        concepts: Extracted concepts with definitions and relevance.
        quiz: Quiz questions with answers and difficulty levels.
        flashcards: Front/back study flashcards.
        mindmap: Hierarchical mind map structure and text rendering.
    """

    question: str
    context_documents: list[Document]
    summary: dict[str, Any]
    concepts: dict[str, Any]
    quiz: dict[str, Any]
    flashcards: dict[str, Any]
    mindmap: dict[str, Any]
