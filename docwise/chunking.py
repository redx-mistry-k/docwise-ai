"""Document chunk generation with stable metadata."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document],
    document_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "document_id": document_id,
                "chunk_index": index,
                "chunk_id": f"{document_id}:{index}",
            }
        )
    return chunks
