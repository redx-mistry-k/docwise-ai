import pytest
from langchain_core.documents import Document

from docwise.chunking import chunk_documents


def test_chunks_keep_citation_metadata_and_stable_ids():
    source = Document(
        page_content="DocWise retrieval evidence. " * 30,
        metadata={"source": "manual.pdf", "page": 7},
    )
    chunks = chunk_documents([source], "abc123", chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    assert chunks[0].metadata["source"] == "manual.pdf"
    assert chunks[0].metadata["page"] == 7
    assert chunks[0].metadata["chunk_id"] == "abc123:0"


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_documents([], "id", chunk_size=100, chunk_overlap=100)
