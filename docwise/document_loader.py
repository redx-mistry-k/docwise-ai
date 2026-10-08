"""Safe loading and validation for PDF, TXT, and Markdown documents."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .exceptions import (
    DocumentTooLargeError,
    EmptyDocumentError,
    InvalidDocumentError,
    UnsupportedDocumentError,
)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def safe_filename(filename: str) -> str:
    """Strip directory components and unsafe characters from an uploaded name."""
    basename = Path(filename.replace("\\", "/")).name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", basename).strip(" .")
    return cleaned[:180] or "document"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_document(path: Path, max_file_size_mb: int = 25) -> None:
    if not path.is_file():
        raise InvalidDocumentError(f"Document does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedDocumentError(f"Unsupported file type. Supported types: {supported}")
    size = path.stat().st_size
    if size == 0:
        raise EmptyDocumentError(f"{path.name} is empty.")
    if size > max_file_size_mb * 1024 * 1024:
        raise DocumentTooLargeError(
            f"{path.name} exceeds the {max_file_size_mb} MB upload limit."
        )


def _load_pdf(path: Path) -> list[Document]:
    try:
        reader = PdfReader(str(path), strict=False)
    except (PdfReadError, OSError, ValueError) as exc:
        raise InvalidDocumentError(f"Could not read PDF '{path.name}': {exc}") from exc

    if reader.is_encrypted:
        try:
            if reader.decrypt("") == 0:
                raise InvalidDocumentError(f"PDF '{path.name}' is password protected.")
        except Exception as exc:
            if isinstance(exc, InvalidDocumentError):
                raise
            raise InvalidDocumentError(f"PDF '{path.name}' is password protected.") from exc

    documents: list[Document] = []
    extraction_errors = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            extraction_errors += 1
            text = ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={"source": path.name, "page": page_number, "file_type": "pdf"},
                )
            )

    if not documents:
        detail = " Text extraction failed on every page." if extraction_errors else ""
        raise EmptyDocumentError(
            f"No extractable text found in '{path.name}'. It may be scanned and require OCR.{detail}"
        )
    return documents


def _load_text(path: Path) -> list[Document]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
        except (UnicodeDecodeError, OSError) as exc:
            raise InvalidDocumentError(f"Could not decode '{path.name}' as text.") from exc
    except OSError as exc:
        raise InvalidDocumentError(f"Could not read '{path.name}': {exc}") from exc
    if not text.strip():
        raise EmptyDocumentError(f"No readable text found in '{path.name}'.")
    return [
        Document(
            page_content=text.strip(),
            metadata={"source": path.name, "page": 1, "file_type": path.suffix.lower().lstrip(".")},
        )
    ]


def load_document(path: Path, max_file_size_mb: int = 25) -> list[Document]:
    """Load a supported file as page-aware LangChain documents."""
    path = path.resolve()
    validate_document(path, max_file_size_mb=max_file_size_mb)
    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)
    return _load_text(path)
