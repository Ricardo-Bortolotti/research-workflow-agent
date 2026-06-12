"""Business logic for document upload and agent workflow analysis."""

import logging
from pathlib import Path
from typing import Any

from api.schemas import (
    AnalysisResultsResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    ConceptItemOutput,
    FlashcardItemOutput,
    MindMapNodeOutput,
    MindMapOutput,
    QuizItemOutput,
    SummaryOutput,
    UploadResponse,
)
from api.storage import AnalysisStorage, StorageError
from graph.workflow import WorkflowError, run_workflow
from rag.embeddings import ChunkEmbedder
from rag.loaders import PDFLoadError, load_pdf
from rag.retriever import DocumentRetriever
from rag.splitter import split_documents
from rag.vectorstore import ChromaVectorStore

logger = logging.getLogger(__name__)


class AnalysisServiceError(Exception):
    """Raised when analysis service operations fail."""


class AnalysisService:
    """Coordinate upload, RAG indexing, and LangGraph agent execution."""

    def __init__(self, storage: AnalysisStorage | None = None) -> None:
        self.storage = storage or AnalysisStorage()

    def upload_pdf(self, filename: str, content: bytes) -> UploadResponse:
        """Save an uploaded PDF file."""
        if not filename.lower().endswith(".pdf"):
            raise AnalysisServiceError("Only PDF files are supported")
        if not content:
            raise AnalysisServiceError("Uploaded file is empty")

        document = self.storage.save_upload(filename, content)
        return UploadResponse(document_id=document.document_id, filename=document.filename)

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        """Index a document, retrieve context, and run the agent workflow."""
        try:
            document = self.storage.get_document(request.document_id)
        except StorageError as exc:
            raise AnalysisServiceError(str(exc)) from exc

        try:
            workflow_result = self._run_pipeline(
                pdf_path=document.file_path,
                document_id=document.document_id,
                question=request.question,
                top_k=request.top_k,
            )
            analysis_id = self.storage.save_results(
                {
                    "document_id": document.document_id,
                    "question": request.question.strip(),
                    "status": "completed",
                    "summary": workflow_result.get("summary"),
                    "concepts": workflow_result.get("concepts"),
                    "quiz": workflow_result.get("quiz"),
                    "flashcards": workflow_result.get("flashcards"),
                    "mindmap": workflow_result.get("mindmap"),
                    "metadata": {
                        "filename": document.filename,
                        "top_k": request.top_k,
                        "chunks_indexed": workflow_result.get("chunks_indexed", 0),
                    },
                }
            )
            return AnalyzeResponse(
                analysis_id=analysis_id,
                document_id=document.document_id,
                question=request.question.strip(),
                status="completed",
                message="Analysis completed successfully",
            )
        except (PDFLoadError, WorkflowError) as exc:
            logger.exception("Analysis failed")
            analysis_id = self.storage.save_results(
                {
                    "document_id": request.document_id,
                    "question": request.question.strip(),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            return AnalyzeResponse(
                analysis_id=analysis_id,
                document_id=request.document_id,
                question=request.question.strip(),
                status="failed",
                message=str(exc),
            )
        except Exception as exc:
            logger.exception("Unexpected analysis failure")
            raise AnalysisServiceError(f"Analysis failed: {exc}") from exc

    def get_results(self, analysis_id: str) -> AnalysisResultsResponse:
        """Return stored analysis results."""
        try:
            payload = self.storage.get_results(analysis_id)
        except StorageError as exc:
            raise AnalysisServiceError(str(exc)) from exc

        return _payload_to_results_response(payload)

    def _run_pipeline(
        self,
        *,
        pdf_path: Path,
        document_id: str,
        question: str,
        top_k: int,
    ) -> dict[str, Any]:
        logger.info("Starting analysis pipeline for document %s", document_id)

        documents = load_pdf(pdf_path)
        chunks = split_documents(documents)

        embedder = ChunkEmbedder()
        vectors = embedder.embed_chunks(chunks)

        collection_name = f"doc_{document_id}"
        store = ChromaVectorStore(collection_name=collection_name)
        store.create_collection(reset=True)
        store.add_documents(chunks, vectors)
        store.persist()

        retriever = DocumentRetriever(
            vector_store=store,
            embedder=embedder,
            top_k=top_k,
        )
        context = retriever.invoke(question)
        workflow_result = run_workflow(question, context)
        workflow_result["chunks_indexed"] = len(chunks)
        return workflow_result


def _payload_to_results_response(payload: dict[str, Any]) -> AnalysisResultsResponse:
    summary_data = payload.get("summary")
    concepts_data = payload.get("concepts", {})
    quiz_data = payload.get("quiz", {})
    flashcards_data = payload.get("flashcards", {})
    mindmap_data = payload.get("mindmap")

    summary = SummaryOutput(**summary_data) if summary_data else None
    concepts = [ConceptItemOutput(**item) for item in concepts_data.get("concepts", [])]
    quiz = [QuizItemOutput(**item) for item in quiz_data.get("questions", [])]
    flashcards = [FlashcardItemOutput(**item) for item in flashcards_data.get("flashcards", [])]

    mindmap = None
    if mindmap_data:
        mindmap = MindMapOutput(
            title=mindmap_data["title"],
            nodes=[_parse_mindmap_node(node) for node in mindmap_data.get("nodes", [])],
            text=mindmap_data.get("text", ""),
        )

    return AnalysisResultsResponse(
        analysis_id=payload["analysis_id"],
        document_id=payload.get("document_id", ""),
        question=payload.get("question", ""),
        status=payload.get("status", "failed"),
        summary=summary,
        concepts=concepts,
        quiz=quiz,
        flashcards=flashcards,
        mindmap=mindmap,
        error=payload.get("error"),
        metadata=payload.get("metadata", {}),
    )


def _parse_mindmap_node(node: dict[str, Any]) -> MindMapNodeOutput:
    return MindMapNodeOutput(
        label=node["label"],
        children=[_parse_mindmap_node(child) for child in node.get("children", [])],
    )
