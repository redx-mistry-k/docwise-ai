"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings with sensible local defaults."""

    project_root: Path
    data_dir: Path
    storage_dir: Path
    chroma_dir: Path
    registry_path: Path
    collection_name: str
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_file_size_mb: int
    retrieval_strategy: str
    min_relevance_score: float
    rerank: bool
    history_aware: bool
    api_timeout_seconds: float
    api_max_retries: int
    model_config_path: Path | None = None
    usage_db_path: Path | None = None
    routing_mode: str = "balanced"
    advanced_rag: bool = True
    query_decomposition: bool = True
    adaptive_retrieval: bool = True
    self_correction: bool = True
    answer_verification: bool = True
    max_retrieval_attempts: int = 2
    max_subqueries: int = 4
    max_adaptive_top_k: int = 10
    max_context_characters: int = 60_000
    max_context_tokens: int = 15_000
    max_additional_llm_calls: int = 0

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or Path.cwd()).resolve()
        load_dotenv(root / ".env")
        storage = Path(os.getenv("DOCWISE_STORAGE_DIR", root / ".docwise")).resolve()
        chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        if chunk_size < 100:
            raise ValueError("CHUNK_SIZE must be at least 100 characters.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE.")

        return cls(
            project_root=root,
            data_dir=Path(os.getenv("DOCWISE_DATA_DIR", root / "data")).resolve(),
            storage_dir=storage,
            chroma_dir=Path(os.getenv("CHROMA_DIR", storage / "chroma")).resolve(),
            registry_path=Path(os.getenv("REGISTRY_PATH", storage / "registry.json")).resolve(),
            collection_name=os.getenv("CHROMA_COLLECTION", "docwise_documents"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            llm_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=max(1, int(os.getenv("TOP_K", "4"))),
            max_file_size_mb=max(1, int(os.getenv("MAX_FILE_SIZE_MB", "25"))),
            retrieval_strategy=os.getenv("RETRIEVAL_STRATEGY", "hybrid").lower(),
            min_relevance_score=float(os.getenv("MIN_RELEVANCE_SCORE", "0.15")),
            rerank=_as_bool(os.getenv("ENABLE_RERANKING"), True),
            history_aware=_as_bool(os.getenv("HISTORY_AWARE"), True),
            api_timeout_seconds=float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60")),
            api_max_retries=max(0, int(os.getenv("GEMINI_MAX_RETRIES", "2"))),
            model_config_path=Path(
                os.getenv("GEMINI_MODEL_CONFIG", root / "config" / "gemini_models.json")
            ).resolve(),
            usage_db_path=Path(
                os.getenv("GEMINI_USAGE_DB", storage / "gemini_usage.sqlite3")
            ).resolve(),
            routing_mode=os.getenv("GEMINI_ROUTING_MODE", "balanced").lower(),
            advanced_rag=_as_bool(os.getenv("ADVANCED_RAG"), True),
            query_decomposition=_as_bool(os.getenv("QUERY_DECOMPOSITION"), True),
            adaptive_retrieval=_as_bool(os.getenv("ADAPTIVE_RETRIEVAL"), True),
            self_correction=_as_bool(os.getenv("SELF_CORRECTING_RAG"), True),
            answer_verification=_as_bool(os.getenv("ANSWER_VERIFICATION"), True),
            max_retrieval_attempts=min(
                2, max(1, int(os.getenv("MAX_RETRIEVAL_ATTEMPTS", "2")))
            ),
            max_subqueries=min(6, max(1, int(os.getenv("MAX_SUBQUERIES", "4")))),
            max_adaptive_top_k=max(1, int(os.getenv("MAX_ADAPTIVE_TOP_K", "10"))),
            max_context_characters=max(
                1_000, int(os.getenv("MAX_CONTEXT_CHARACTERS", "60000"))
            ),
            max_context_tokens=max(
                250, int(os.getenv("MAX_CONTEXT_TOKENS", "15000"))
            ),
            max_additional_llm_calls=min(
                1, max(0, int(os.getenv("MAX_ADDITIONAL_LLM_CALLS", "0")))
            ),
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "documents").mkdir(parents=True, exist_ok=True)
