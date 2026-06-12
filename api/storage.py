"""File-based persistence for uploads and analysis results."""

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredDocument:
    """Metadata for an uploaded PDF."""

    document_id: str
    filename: str
    file_path: Path


class StorageError(Exception):
    """Raised when storage operations fail."""


class AnalysisStorage:
    """Persist uploaded documents and analysis results on disk."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.uploads_dir = self.data_dir / "metadata" / "uploads"
        self.results_dir = self.data_dir / "metadata" / "results"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, filename: str, content: bytes) -> StoredDocument:
        """Save an uploaded PDF and its metadata."""
        document_id = uuid.uuid4().hex
        safe_name = Path(filename).name
        file_path = self.raw_dir / f"{document_id}.pdf"
        file_path.write_bytes(content)

        metadata = {
            "document_id": document_id,
            "filename": safe_name,
            "file_path": str(file_path.resolve()),
        }
        self._write_json(self.uploads_dir / f"{document_id}.json", metadata)
        logger.info("Saved upload: %s (%s)", document_id, safe_name)
        return StoredDocument(document_id=document_id, filename=safe_name, file_path=file_path)

    def get_document(self, document_id: str) -> StoredDocument:
        """Load metadata for a previously uploaded document."""
        metadata_path = self.uploads_dir / f"{document_id}.json"
        if not metadata_path.exists():
            raise StorageError(f"Document not found: {document_id}")

        metadata = self._read_json(metadata_path)
        file_path = Path(metadata["file_path"])
        if not file_path.exists():
            raise StorageError(f"PDF file missing for document: {document_id}")

        return StoredDocument(
            document_id=metadata["document_id"],
            filename=metadata["filename"],
            file_path=file_path,
        )

    def save_results(self, payload: dict[str, Any]) -> str:
        """Persist analysis results and return the analysis ID."""
        analysis_id = payload.get("analysis_id") or uuid.uuid4().hex
        payload["analysis_id"] = analysis_id
        self._write_json(self.results_dir / f"{analysis_id}.json", payload)
        logger.info("Saved analysis results: %s", analysis_id)
        return analysis_id

    def get_results(self, analysis_id: str) -> dict[str, Any]:
        """Load analysis results by ID."""
        results_path = self.results_dir / f"{analysis_id}.json"
        if not results_path.exists():
            raise StorageError(f"Analysis not found: {analysis_id}")
        return self._read_json(results_path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise StorageError(f"Invalid JSON object in {path}")
        return data

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
