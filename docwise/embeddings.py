"""Embedding model factory."""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=3)
def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Load and cache normalized local sentence-transformer embeddings."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
