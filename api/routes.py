"""FastAPI routes for the Book Research Agent."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.schemas import (
    AnalysisResultsResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
    UploadResponse,
)
from api.service import AnalysisService, AnalysisServiceError

router = APIRouter()


def get_analysis_service() -> AnalysisService:
    """Dependency provider for AnalysisService."""
    return AnalysisService()


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Upload a PDF document",
)
async def upload_pdf(
    file: UploadFile = File(...),
    service: AnalysisService = Depends(get_analysis_service),
) -> UploadResponse:
    """Upload a PDF to be analyzed later."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required")

    content = await file.read()
    try:
        return service.upload_pdf(file.filename, content)
    except AnalysisServiceError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    summary="Analyze an uploaded document",
)
async def analyze_document(
    request: AnalyzeRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalyzeResponse:
    """Index the document, run RAG retrieval, and execute the agent workflow."""
    try:
        return service.analyze(request)
    except AnalysisServiceError as exc:
        if "not found" in str(exc).lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/results/{analysis_id}",
    response_model=AnalysisResultsResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get analysis results",
)
async def get_results(
    analysis_id: str,
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResultsResponse:
    """Retrieve stored results for a completed or failed analysis."""
    try:
        return service.get_results(analysis_id)
    except AnalysisServiceError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
