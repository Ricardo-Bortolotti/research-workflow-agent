"""Tests for PDF document loaders."""

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from rag.loaders import PDFLoadError, load_pdf


def test_load_pdf_returns_documents(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    mock_documents = [
        Document(page_content="Page 1", metadata={"source": str(pdf_path), "page": 0}),
        Document(page_content="Page 2", metadata={"source": str(pdf_path), "page": 1}),
    ]

    with patch("rag.loaders.PyPDFLoader") as mock_loader_cls:
        mock_loader_cls.return_value.load.return_value = mock_documents
        result = load_pdf(pdf_path)

    assert result == mock_documents
    mock_loader_cls.assert_called_once_with(str(pdf_path.resolve()))


def test_load_pdf_file_not_found() -> None:
    with pytest.raises(PDFLoadError, match="File not found"):
        load_pdf("/nonexistent/document.pdf")


def test_load_pdf_unsupported_extension(tmp_path: Path) -> None:
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("not a pdf")

    with pytest.raises(PDFLoadError, match="Unsupported file extension"):
        load_pdf(txt_path)


def test_load_pdf_empty_document_list(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    with patch("rag.loaders.PyPDFLoader") as mock_loader_cls:
        mock_loader_cls.return_value.load.return_value = []
        with pytest.raises(PDFLoadError, match="no extractable pages"):
            load_pdf(pdf_path)


def test_load_pdf_parser_failure(tmp_path: Path) -> None:
    pdf_path = tmp_path / "corrupt.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    with patch("rag.loaders.PyPDFLoader") as mock_loader_cls:
        mock_loader_cls.return_value.load.side_effect = RuntimeError("invalid pdf structure")
        with pytest.raises(PDFLoadError, match="Failed to parse PDF"):
            load_pdf(pdf_path)
