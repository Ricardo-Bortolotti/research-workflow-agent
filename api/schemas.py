"""Pydantic schemas for the Book Research Agent API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response after a PDF upload."""

    document_id: str
    filename: str
    message: str = "PDF uploaded successfully"


class AnalyzeRequest(BaseModel):
    """Request body to analyze an uploaded document."""

    document_id: str = Field(..., description="ID returned by POST /upload")
    question: str = Field(..., min_length=1, description="Research question for the agents")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class AnalyzeResponse(BaseModel):
    """Response after starting or completing an analysis."""

    analysis_id: str
    document_id: str
    question: str
    status: Literal["completed", "failed"]
    message: str


class SummaryOutput(BaseModel):
    """Executive summary and key insights."""

    executive_summary: str
    key_insights: list[str]


class ConceptItemOutput(BaseModel):
    """A single extracted concept."""

    concept: str
    definition: str
    relevance: str


class QuizItemOutput(BaseModel):
    """A single quiz question."""

    question: str
    answer: str
    difficulty: str


class FlashcardItemOutput(BaseModel):
    """A single flashcard."""

    front: str
    back: str


class MindMapNodeOutput(BaseModel):
    """A node in the mind map tree."""

    label: str
    children: list["MindMapNodeOutput"] = Field(default_factory=list)


class MindMapOutput(BaseModel):
    """Hierarchical mind map output."""

    title: str
    nodes: list[MindMapNodeOutput]
    text: str


class AnalysisResultsResponse(BaseModel):
    """Full analysis results returned by GET /results."""

    analysis_id: str
    document_id: str
    question: str
    status: Literal["completed", "failed"]
    summary: SummaryOutput | None = None
    concepts: list[ConceptItemOutput] = Field(default_factory=list)
    quiz: list[QuizItemOutput] = Field(default_factory=list)
    flashcards: list[FlashcardItemOutput] = Field(default_factory=list)
    mindmap: MindMapOutput | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard API error response."""

    detail: str


MindMapNodeOutput.model_rebuild()
