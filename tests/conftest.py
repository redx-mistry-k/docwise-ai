from __future__ import annotations

import hashlib
import math

import pytest
from langchain_core.embeddings import Embeddings

from docwise.config import Settings


class DeterministicEmbeddings(Embeddings):
    """Small offline embedding model for repeatable storage tests."""

    dimensions = 32

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            slot = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % self.dimensions
            vector[slot] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def embeddings():
    return DeterministicEmbeddings()


@pytest.fixture
def settings(tmp_path):
    base = Settings.from_env(tmp_path)
    return Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        storage_dir=tmp_path / ".docwise",
        chroma_dir=tmp_path / ".docwise" / "chroma",
        registry_path=tmp_path / ".docwise" / "registry.json",
        collection_name="test_documents",
        embedding_model=base.embedding_model,
        llm_model=base.llm_model,
        chunk_size=200,
        chunk_overlap=30,
        top_k=4,
        max_file_size_mb=2,
        retrieval_strategy="hybrid",
        min_relevance_score=0.1,
        rerank=True,
        history_aware=True,
        api_timeout_seconds=10,
        api_max_retries=0,
    )
