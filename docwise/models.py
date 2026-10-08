"""Typed data models shared across the application."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.documents import Document


@dataclass(slots=True)
class DocumentRecord:
    document_id: str
    filename: str
    checksum: str
    file_type: str
    page_count: int
    chunk_count: int
    indexed_at: str
    stored_path: str
    chunk_size: int
    chunk_overlap: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocumentRecord":
        return cls(**value)


@dataclass(slots=True)
class IndexResult:
    record: DocumentRecord
    status: str
    message: str


@dataclass(slots=True)
class RetrievedChunk:
    document: Document
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    retrieval_reason: str = "relevance"

    @property
    def citation(self) -> str:
        source = self.document.metadata.get("source", "Unknown")
        page = self.document.metadata.get("page", 1)
        return f"{source}, page {page}"


@dataclass(slots=True)
class RAGResponse:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    rewritten_question: str | None = None
    insufficient_context: bool = False
    diagnostics: "RetrievalDiagnostics | None" = None
    model_name: str | None = None
    fallback_reason: str | None = None


@dataclass(slots=True)
class RetrievalDiagnostics:
    query: str
    strategy: str
    comparison_mode: bool
    total_candidates: int
    selected_count: int
    per_document: dict[str, int] = field(default_factory=dict)
    context_characters: int = 0
    truncated: bool = False
    omitted_chunks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    context_tokens: int = 0
    complexity: str = "simple"
    subqueries: list[str] = field(default_factory=list)
    subquery_matches: dict[str, int] = field(default_factory=dict)
    adaptive_top_k: int = 0
    retrieval_attempts: int = 1
    additional_retrieval_attempts: int = 0
    reformulated_query: str | None = None
    llm_calls: int = 0
    verification_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: list[str] = field(default_factory=list)
    uncertain_claims: list[str] = field(default_factory=list)
    contradictory_claims: list[str] = field(default_factory=list)
    evidence_contradictions: list[str] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    citation_correctness: float | None = None
