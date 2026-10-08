from pathlib import Path

import pytest

from docwise.exceptions import DocumentNotFoundError
from docwise.retrieval import RetrievalOptions, Retriever
from docwise.vector_store import VectorStoreManager


def test_index_duplicate_retrieve_reindex_and_delete(settings, embeddings, tmp_path: Path):
    path = tmp_path / "benefits.txt"
    path.write_text(
        ("Employees receive twenty days of annual leave. "
         "Parental leave and health insurance are included. ") * 8,
        encoding="utf-8",
    )
    manager = VectorStoreManager(settings, embeddings=embeddings)
    first = manager.index_file(path)
    duplicate = manager.index_file(path)
    assert first.status == "indexed"
    assert duplicate.status == "duplicate"
    assert manager.stats()["document_count"] == 1
    assert manager.stats()["chunk_count"] == first.record.chunk_count

    results = Retriever(manager).retrieve(
        "How much annual leave do employees receive?",
        RetrievalOptions(strategy="hybrid", top_k=2, rerank=True),
    )
    assert results
    assert results[0].document.metadata["source"] == "benefits.txt"
    assert results[0].document.metadata["page"] == 1
    assert 0 <= results[0].score <= 1

    reindexed = manager.reindex_document(first.record.document_id, chunk_size=150)
    assert reindexed.record.filename == "benefits.txt"
    assert reindexed.record.chunk_size == 150
    deleted_chunks = manager.delete_document(first.record.document_id)
    assert deleted_chunks > 0
    assert manager.list_documents() == []
    with pytest.raises(DocumentNotFoundError):
        manager.delete_document(first.record.document_id)
