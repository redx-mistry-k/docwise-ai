"""Persistent Chroma storage and document lifecycle management."""

from __future__ import annotations

import json
import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from .chunking import chunk_documents
from .config import Settings
from .document_loader import load_document, safe_filename, sha256_file
from .embeddings import get_embeddings
from .exceptions import DocumentNotFoundError, EmptyDocumentError
from .models import DocumentRecord, IndexResult

logger = logging.getLogger(__name__)


class DocumentRegistry:
    """Small JSON catalog kept alongside Chroma for document-level operations."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Document registry is unreadable: {self.path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Document registry has an invalid format: {self.path}")
        return payload

    def _write(self, payload: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            records = [DocumentRecord.from_dict(item) for item in self._read().values()]
        return sorted(records, key=lambda record: record.indexed_at, reverse=True)

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            value = self._read().get(document_id)
        return DocumentRecord.from_dict(value) if value else None

    def find_by_checksum(self, checksum: str) -> DocumentRecord | None:
        return next((record for record in self.list() if record.checksum == checksum), None)

    def upsert(self, record: DocumentRecord) -> None:
        with self._lock:
            payload = self._read()
            payload[record.document_id] = record.to_dict()
            self._write(payload)

    def remove(self, document_id: str) -> None:
        with self._lock:
            payload = self._read()
            payload.pop(document_id, None)
            self._write(payload)


class VectorStoreManager:
    """Owns the vector index and keeps it consistent with the document registry."""

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings | None = None,
        vectorstore: Chroma | None = None,
    ):
        self.settings = settings
        self.settings.ensure_directories()
        self.registry = DocumentRegistry(settings.registry_path)
        self.embeddings = embeddings or get_embeddings(settings.embedding_model)
        self.vectorstore = vectorstore or Chroma(
            collection_name=settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(settings.chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._lock = threading.RLock()

    @property
    def documents_dir(self) -> Path:
        return self.settings.storage_dir / "documents"

    def list_documents(self) -> list[DocumentRecord]:
        return self.registry.list()

    def index_file(
        self,
        path: Path,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        force: bool = False,
        copy_to_storage: bool = True,
        filename_override: str | None = None,
    ) -> IndexResult:
        """Validate, chunk, embed, and persist one file without duplicate indexing."""
        path = path.resolve()
        checksum = sha256_file(path)
        with self._lock:
            duplicate = self.registry.find_by_checksum(checksum)
            if duplicate and not force:
                return IndexResult(
                    record=duplicate,
                    status="duplicate",
                    message=f"'{path.name}' is already indexed as '{duplicate.filename}'.",
                )

            pages = load_document(path, max_file_size_mb=self.settings.max_file_size_mb)
            document_id = checksum[:24]
            size = chunk_size or self.settings.chunk_size
            overlap = self.settings.chunk_overlap if chunk_overlap is None else chunk_overlap
            chunks = chunk_documents(pages, document_id, size, overlap)
            if not chunks:
                raise EmptyDocumentError(f"No chunks could be created from '{path.name}'.")

            filename = safe_filename(filename_override or path.name)
            stored_name = f"{document_id}_{filename}"
            destination = self.documents_dir / stored_name
            copied = False
            if copy_to_storage and path != destination.resolve():
                shutil.copy2(path, destination)
                copied = True
            elif not destination.exists():
                destination = path

            ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
            if force and duplicate:
                self._delete_vectors(duplicate.document_id)
            try:
                self.vectorstore.add_documents(chunks, ids=ids)
                record = DocumentRecord(
                    document_id=document_id,
                    filename=filename,
                    checksum=checksum,
                    file_type=path.suffix.lower().lstrip("."),
                    page_count=len(pages),
                    chunk_count=len(chunks),
                    indexed_at=datetime.now(timezone.utc).isoformat(),
                    stored_path=destination.name if destination.parent == self.documents_dir else str(destination),
                    chunk_size=size,
                    chunk_overlap=overlap,
                )
                self.registry.upsert(record)
            except Exception:
                logger.exception("Failed to index document %s", filename)
                self.vectorstore.delete(ids=ids)
                if copied and destination.exists():
                    destination.unlink()
                raise
            return IndexResult(
                record=record,
                status="indexed",
                message=f"Indexed '{filename}' into {len(chunks)} chunks.",
            )

    def index_directory(
        self,
        directory: Path,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        force: bool = False,
    ) -> list[IndexResult]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {directory}")
        paths = sorted(
            path for path in directory.iterdir() if path.suffix.lower() in {".pdf", ".txt", ".md", ".markdown"}
        )
        return [
            self.index_file(
                path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                force=force,
            )
            for path in paths
        ]

    def _stored_file(self, record: DocumentRecord) -> Path:
        path = Path(record.stored_path)
        return path if path.is_absolute() else self.documents_dir / path

    def _delete_vectors(self, document_id: str) -> int:
        payload = self.vectorstore.get(where={"document_id": document_id}, include=[])
        ids = payload.get("ids", [])
        if ids:
            self.vectorstore.delete(ids=ids)
        return len(ids)

    def delete_document(self, document_id: str, *, delete_file: bool = True) -> int:
        with self._lock:
            record = self.registry.get(document_id)
            if not record:
                raise DocumentNotFoundError(f"Indexed document not found: {document_id}")
            deleted = self._delete_vectors(document_id)
            self.registry.remove(document_id)
            stored_file = self._stored_file(record)
            if delete_file and stored_file.exists():
                try:
                    stored_file.resolve().relative_to(self.documents_dir.resolve())
                except ValueError:
                    logger.warning("Refusing to delete file outside managed storage: %s", stored_file)
                else:
                    stored_file.unlink()
            return deleted

    def reindex_document(
        self,
        document_id: str,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IndexResult:
        record = self.registry.get(document_id)
        if not record:
            raise DocumentNotFoundError(f"Indexed document not found: {document_id}")
        stored_file = self._stored_file(record)
        if not stored_file.exists():
            raise DocumentNotFoundError(
                f"Stored copy for '{record.filename}' is missing; upload it again to reindex."
            )
        return self.index_file(
            stored_file,
            chunk_size=chunk_size or record.chunk_size,
            chunk_overlap=record.chunk_overlap if chunk_overlap is None else chunk_overlap,
            force=True,
            copy_to_storage=False,
            filename_override=record.filename,
        )

    def all_chunks(self) -> list[dict[str, Any]]:
        payload = self.vectorstore.get(include=["documents", "metadatas"])
        ids = payload.get("ids", [])
        documents = payload.get("documents", []) or []
        metadatas = payload.get("metadatas", []) or []
        return [
            {"id": chunk_id, "text": text, "metadata": metadata or {}}
            for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=False)
        ]

    def stats(self) -> dict[str, Any]:
        documents = self.list_documents()
        return {
            "document_count": len(documents),
            "chunk_count": sum(record.chunk_count for record in documents),
            "embedding_model": self.settings.embedding_model,
            "llm_model": self.settings.llm_model,
        }
