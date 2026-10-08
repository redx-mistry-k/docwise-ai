"""Vector, keyword, and hybrid retrieval strategies."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from langchain_core.documents import Document

from .models import RetrievalDiagnostics, RetrievedChunk
from .vector_store import VectorStoreManager

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


@dataclass(slots=True)
class RetrievalOptions:
    strategy: str = "hybrid"
    top_k: int = 4
    rerank: bool = True
    vector_weight: float = 0.65
    per_document_k: int = 2
    short_document_chunks: int = 8
    comparison_max_chunks: int = 24
    comparison_max_documents: int = 4
    max_context_characters: int = 60_000
    max_context_tokens: int = 15_000
    advanced: bool = True
    decompose: bool = True
    adaptive: bool = True
    self_correct: bool = True
    verify_answer: bool = True
    max_subqueries: int = 4
    max_adaptive_top_k: int = 10
    max_retrieval_attempts: int = 2


class Retriever:
    def __init__(self, store: VectorStoreManager):
        self.store = store
        self._raw_chunks_cache: list[dict] | None = None
        self._tokenized_cache: list[list[str]] | None = None
        self._document_frequency_cache: Counter | None = None
        self._average_length_cache: float | None = None
        self._has_documents_cache: bool | None = None

    def invalidate_cache(self) -> None:
        """Refresh retrieval snapshots after an explicit index mutation."""
        self._raw_chunks_cache = None
        self._tokenized_cache = None
        self._document_frequency_cache = None
        self._average_length_cache = None
        self._has_documents_cache = None

    def _all_chunks(self) -> list[dict]:
        if self._raw_chunks_cache is None:
            self._raw_chunks_cache = self.store.all_chunks()
        return self._raw_chunks_cache

    def _vector(self, query: str, limit: int) -> list[RetrievedChunk]:
        if self._has_documents_cache is None:
            self._has_documents_cache = bool(self.store.list_documents())
        if not self._has_documents_cache:
            return []
        matches = self.store.vectorstore.similarity_search_with_score(query, k=limit)
        return [
            RetrievedChunk(
                document=document,
                score=max(0.0, min(1.0, 1.0 - float(distance))),
                vector_score=max(0.0, min(1.0, 1.0 - float(distance))),
            )
            for document, distance in matches
        ]

    def _keyword(self, query: str, limit: int) -> list[RetrievedChunk]:
        chunks = self._all_chunks()
        query_terms = tokenize(query)
        if not chunks or not query_terms:
            return []
        if self._tokenized_cache is None:
            self._tokenized_cache = [tokenize(chunk["text"]) for chunk in chunks]
            self._document_frequency_cache = Counter(
                term for tokens in self._tokenized_cache for term in set(tokens)
            )
            self._average_length_cache = sum(map(len, self._tokenized_cache)) / max(
                len(self._tokenized_cache), 1
            )
        tokenized = self._tokenized_cache
        document_frequency = self._document_frequency_cache or Counter()
        average_length = self._average_length_cache or 1.0
        raw_scores: list[float] = []
        for tokens in tokenized:
            counts = Counter(tokens)
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                inverse_frequency = math.log(
                    1 + (len(tokenized) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
                )
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / max(average_length, 1))
                score += inverse_frequency * frequency * 2.5 / denominator
            raw_scores.append(score)
        normalized = _normalize(raw_scores)
        ranked = sorted(
            zip(chunks, normalized, strict=False), key=lambda item: item[1], reverse=True
        )
        return [
            RetrievedChunk(
                document=Document(page_content=chunk["text"], metadata=chunk["metadata"]),
                score=score,
                keyword_score=score,
            )
            for chunk, score in ranked[:limit]
            if score > 0
        ]

    def _rank(
        self,
        query: str,
        options: RetrievalOptions,
        candidate_limit: int,
    ) -> list[RetrievedChunk]:
        strategy = options.strategy.lower()
        if strategy not in {"vector", "keyword", "hybrid"}:
            raise ValueError("Retrieval strategy must be vector, keyword, or hybrid.")
        if strategy == "vector":
            results = self._vector(query, candidate_limit)
        elif strategy == "keyword":
            results = self._keyword(query, candidate_limit)
        else:
            vector_results = self._vector(query, candidate_limit)
            keyword_results = self._keyword(query, candidate_limit)
            merged: dict[str, RetrievedChunk] = {}
            for item in vector_results:
                key = str(item.document.metadata.get("chunk_id", item.document.page_content))
                merged[key] = item
            for item in keyword_results:
                key = str(item.document.metadata.get("chunk_id", item.document.page_content))
                if key in merged:
                    merged[key].keyword_score = item.keyword_score
                else:
                    merged[key] = item
            for item in merged.values():
                item.score = (
                    options.vector_weight * item.vector_score
                    + (1 - options.vector_weight) * item.keyword_score
                )
            results = list(merged.values())

        if options.rerank:
            query_terms = set(tokenize(query))
            for item in results:
                chunk_terms = set(tokenize(item.document.page_content))
                coverage = len(query_terms & chunk_terms) / max(len(query_terms), 1)
                item.score = 0.85 * item.score + 0.15 * coverage
        return sorted(results, key=lambda item: item.score, reverse=True)

    def retrieve(self, query: str, options: RetrievalOptions | None = None) -> list[RetrievedChunk]:
        options = options or RetrievalOptions()
        candidate_limit = max(options.top_k * 3, options.top_k)
        return self._rank(query, options, candidate_limit)[: options.top_k]

    def retrieve_with_diagnostics(
        self,
        query: str,
        options: RetrievalOptions | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalDiagnostics]:
        options = options or RetrievalOptions()
        chunks = self.retrieve(query, options)
        all_chunks = self._all_chunks()
        return chunks, self._diagnostics(query, options, chunks, len(all_chunks), False, [])

    def retrieve_intelligently(
        self,
        query: str,
        subqueries: list[str],
        options: RetrievalOptions | None = None,
        *,
        comparison_mode: bool = False,
        complexity: str = "simple",
        adaptive_top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalDiagnostics]:
        """Retrieve a bounded, deduplicated evidence set across focused subqueries."""
        options = options or RetrievalOptions()
        queries = list(dict.fromkeys(value.strip() for value in subqueries if value.strip()))
        queries = queries[: options.max_subqueries] or [query]
        target_k = max(
            options.top_k,
            min(options.max_adaptive_top_k, adaptive_top_k or options.top_k),
        )
        adaptive_options = RetrievalOptions(
            strategy=options.strategy,
            top_k=target_k,
            rerank=options.rerank,
            vector_weight=options.vector_weight,
            per_document_k=options.per_document_k,
            short_document_chunks=options.short_document_chunks,
            comparison_max_chunks=options.comparison_max_chunks,
            comparison_max_documents=options.comparison_max_documents,
            max_context_characters=options.max_context_characters,
            max_context_tokens=options.max_context_tokens,
            advanced=options.advanced,
            decompose=options.decompose,
            adaptive=options.adaptive,
            self_correct=options.self_correct,
            verify_answer=options.verify_answer,
            max_subqueries=options.max_subqueries,
            max_adaptive_top_k=options.max_adaptive_top_k,
            max_retrieval_attempts=options.max_retrieval_attempts,
        )

        if comparison_mode:
            selected, diagnostics = self.retrieve_for_comparison(query, adaptive_options)
            base_omitted = list(diagnostics.omitted_chunks)
        else:
            selected = []
            base_omitted = []
            diagnostics = self._diagnostics(
                query,
                adaptive_options,
                [],
                len(self._all_chunks()),
                False,
                [],
            )

        merged: dict[str, RetrievedChunk] = {
            _chunk_id(item): item for item in selected
        }
        subquery_matches: dict[str, int] = {}
        subquery_ids: dict[str, set[str]] = {}
        per_query_k = max(2, math.ceil(target_k / max(len(queries), 1)))
        for position, subquery in enumerate(queries, start=1):
            ranked = self._rank(
                subquery,
                adaptive_options,
                max(target_k * 3, per_query_k),
            )
            relevant = [item for item in ranked if item.score > 0][:per_query_k]
            subquery_ids[subquery] = {_chunk_id(item) for item in relevant}
            for item in relevant:
                key = _chunk_id(item)
                reason = f"focused query {position}"
                if key in merged:
                    merged[key].score = max(merged[key].score, item.score)
                    if reason not in merged[key].retrieval_reason:
                        merged[key].retrieval_reason += f"; {reason}"
                else:
                    item.retrieval_reason = reason
                    merged[key] = item

        ranked_merged = (
            list(merged.values())
            if comparison_mode
            else sorted(merged.values(), key=lambda item: item.score, reverse=True)
        )
        limit = options.comparison_max_chunks if comparison_mode else target_k
        selected, omitted = _budget_chunks(ranked_merged, adaptive_options, limit)
        omitted = list(dict.fromkeys([*base_omitted, *omitted]))
        selected_ids = {_chunk_id(item) for item in selected}
        subquery_matches = {
            subquery: len(ids & selected_ids) for subquery, ids in subquery_ids.items()
        }
        if not comparison_mode:
            selected.sort(key=lambda item: item.score, reverse=True)
        diagnostics = self._diagnostics(
            query,
            adaptive_options,
            selected,
            len(self._all_chunks()),
            comparison_mode,
            omitted,
        )
        diagnostics.complexity = complexity
        diagnostics.subqueries = queries
        diagnostics.subquery_matches = subquery_matches
        diagnostics.adaptive_top_k = target_k
        return selected, diagnostics

    def retrieve_for_comparison(
        self,
        query: str,
        options: RetrievalOptions | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalDiagnostics]:
        """Retrieve comprehensive evidence while guaranteeing representation per document."""
        options = options or RetrievalOptions()
        raw_chunks = self._all_chunks()
        if not raw_chunks:
            diagnostics = self._diagnostics(query, options, [], 0, True, [])
            return [], diagnostics

        ranked = self._rank(query, options, len(raw_chunks))
        ranked_by_id = {
            str(item.document.metadata.get("chunk_id")): item for item in ranked
        }
        by_document: dict[str, list[RetrievedChunk]] = {}
        for raw in raw_chunks:
            chunk_id = str(raw["metadata"].get("chunk_id", raw["id"]))
            item = ranked_by_id.get(chunk_id)
            if item is None:
                item = RetrievedChunk(
                    document=Document(page_content=raw["text"], metadata=raw["metadata"]),
                    score=0.0,
                    retrieval_reason="document coverage",
                )
            source = str(item.document.metadata.get("source", "Unknown"))
            by_document.setdefault(source, []).append(item)

        document_scores = sorted(
            (
                (source, max((item.score for item in document_chunks), default=0.0))
                for source, document_chunks in by_document.items()
            ),
            key=lambda value: value[1],
            reverse=True,
        )
        best_score = document_scores[0][1] if document_scores else 0.0
        score_floor = max(0.05, best_score * 0.25)
        query_tokens = set(tokenize(query))
        forced_documents = [
            source
            for source, _ in document_scores
            if _source_matches_query(source, query_tokens)
        ]
        routed_documents = list(dict.fromkeys([
            *forced_documents,
            *[
            source for source, score in document_scores if score >= score_floor
            ],
        ]))[: options.comparison_max_documents]
        if not routed_documents and document_scores:
            routed_documents = [document_scores[0][0]]
        by_document = {
            source: by_document[source] for source in routed_documents
        }

        selected: list[RetrievedChunk] = []
        selected_ids: set[str] = set()

        def add(item: RetrievedChunk, reason: str) -> None:
            chunk_id = str(item.document.metadata.get("chunk_id", item.document.page_content))
            if chunk_id in selected_ids:
                return
            item.retrieval_reason = reason
            selected.append(item)
            selected_ids.add(chunk_id)

        for document_chunks in by_document.values():
            document_chunks.sort(
                key=lambda item: int(item.document.metadata.get("chunk_index", 0))
            )
            if len(document_chunks) <= options.short_document_chunks:
                for item in document_chunks:
                    add(item, "complete short-document coverage")
            else:
                for item in sorted(document_chunks, key=lambda value: value.score, reverse=True)[
                    : options.per_document_k
                ]:
                    add(item, "per-document relevance coverage")

        for item in ranked:
            if len(selected) >= options.comparison_max_chunks:
                break
            add(item, "global relevance")

        selected.sort(
            key=lambda item: (
                str(item.document.metadata.get("source", "")),
                int(item.document.metadata.get("chunk_index", 0)),
            )
        )
        selected_by_document: dict[str, list[RetrievedChunk]] = {}
        for item in selected:
            source = str(item.document.metadata.get("source", "Unknown"))
            selected_by_document.setdefault(source, []).append(item)
        round_robin: list[RetrievedChunk] = []
        longest = max(map(len, selected_by_document.values()), default=0)
        for position in range(longest):
            for document_chunks in selected_by_document.values():
                if position < len(document_chunks):
                    round_robin.append(document_chunks[position])

        budgeted: list[RetrievedChunk] = []
        omitted: list[str] = []
        used_characters = 0
        used_tokens = 0
        for item in round_robin:
            chunk_id = str(item.document.metadata.get("chunk_id", "unknown"))
            size = len(item.document.page_content)
            tokens = _estimate_tokens(item.document.page_content)
            if (
                len(budgeted) >= options.comparison_max_chunks
                or used_characters + size > options.max_context_characters
                or used_tokens + tokens > options.max_context_tokens
            ):
                omitted.append(chunk_id)
                continue
            budgeted.append(item)
            used_characters += size
            used_tokens += tokens

        budgeted.sort(key=lambda item: item.score, reverse=True)

        diagnostics = self._diagnostics(
            query,
            options,
            budgeted,
            len(raw_chunks),
            True,
            omitted,
        )
        if len(routed_documents) < len(document_scores):
            diagnostics.warnings.append(
                f"Document routing selected {len(routed_documents)} of "
                f"{len(document_scores)} indexed documents for comparison coverage."
            )
        return budgeted, diagnostics

    @staticmethod
    def _diagnostics(
        query: str,
        options: RetrievalOptions,
        chunks: list[RetrievedChunk],
        total_candidates: int,
        comparison_mode: bool,
        omitted: list[str],
    ) -> RetrievalDiagnostics:
        per_document = Counter(
            str(item.document.metadata.get("source", "Unknown")) for item in chunks
        )
        return RetrievalDiagnostics(
            query=query,
            strategy=options.strategy,
            comparison_mode=comparison_mode,
            total_candidates=total_candidates,
            selected_count=len(chunks),
            per_document=dict(per_document),
            context_characters=sum(len(item.document.page_content) for item in chunks),
            context_tokens=sum(_estimate_tokens(item.document.page_content) for item in chunks),
            truncated=bool(omitted),
            omitted_chunks=omitted,
            adaptive_top_k=options.top_k,
        )


def merge_retrieval_results(
    primary: list[RetrievedChunk],
    additional: list[RetrievedChunk],
    options: RetrievalOptions,
    *,
    comparison_mode: bool,
) -> tuple[list[RetrievedChunk], list[str]]:
    """Combine a bounded retry with initial evidence without duplicating chunks."""
    merged: dict[str, RetrievedChunk] = {}
    for item in [*primary, *additional]:
        key = _chunk_id(item)
        if key not in merged or item.score > merged[key].score:
            merged[key] = item
    ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    limit = options.comparison_max_chunks if comparison_mode else options.max_adaptive_top_k
    return _budget_chunks(ranked, options, limit)


def _budget_chunks(
    chunks: list[RetrievedChunk],
    options: RetrievalOptions,
    limit: int,
) -> tuple[list[RetrievedChunk], list[str]]:
    selected: list[RetrievedChunk] = []
    omitted: list[str] = []
    characters = 0
    tokens = 0
    for item in chunks:
        size = len(item.document.page_content)
        estimated_tokens = _estimate_tokens(item.document.page_content)
        if (
            len(selected) >= limit
            or characters + size > options.max_context_characters
            or tokens + estimated_tokens > options.max_context_tokens
        ):
            omitted.append(_chunk_id(item))
            continue
        selected.append(item)
        characters += size
        tokens += estimated_tokens
    return selected, omitted


def _chunk_id(item: RetrievedChunk) -> str:
    return str(item.document.metadata.get("chunk_id", item.document.page_content))


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _source_matches_query(source: str, query_tokens: set[str]) -> bool:
    source_tokens = set(tokenize(source.replace("_", " ").replace(".", " ")))
    if source_tokens & query_tokens:
        return True
    lowered = source.lower()
    if {"resume", "cv"} & query_tokens and ("resume" in lowered or "cv" in source_tokens):
        return True
    if "job" in query_tokens and "description" in query_tokens:
        return "jd" in source_tokens or "job" in source_tokens
    return False
