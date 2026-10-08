from pathlib import Path

import pytest
from pypdf import PdfWriter

from docwise.document_loader import load_document, safe_filename
from docwise.exceptions import EmptyDocumentError, InvalidDocumentError, UnsupportedDocumentError


def test_load_text_adds_page_level_metadata(tmp_path: Path):
    path = tmp_path / "guide.md"
    path.write_text("# Guide\nDocWise supports Markdown.", encoding="utf-8")
    documents = load_document(path)
    assert len(documents) == 1
    assert documents[0].metadata == {"source": "guide.md", "page": 1, "file_type": "md"}


def test_empty_pdf_reports_scanned_or_empty_content(tmp_path: Path):
    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with path.open("wb") as handle:
        writer.write(handle)
    with pytest.raises(EmptyDocumentError, match="scanned"):
        load_document(path)


def test_corrupt_pdf_is_rejected(tmp_path: Path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(InvalidDocumentError):
        load_document(path)


def test_unsupported_file_and_filename_sanitization(tmp_path: Path):
    path = tmp_path / "payload.exe"
    path.write_bytes(b"unsafe")
    with pytest.raises(UnsupportedDocumentError):
        load_document(path)
    assert safe_filename("../../quarterly:report?.pdf") == "quarterly_report_.pdf"
