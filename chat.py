"""Interactive terminal client for DocWise AI."""

from __future__ import annotations

import argparse

from docwise.config import Settings
from docwise.exceptions import DocWiseError
from docwise.history import ConversationHistory
from docwise.rag import RAGService
from docwise.retrieval import RetrievalOptions, Retriever
from docwise.vector_store import VectorStoreManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with indexed documents")
    parser.add_argument("--strategy", choices=["vector", "keyword", "hybrid"], default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--basic-rag", action="store_true", help="Disable advanced RAG checks")
    args = parser.parse_args()

    settings = Settings.from_env()
    manager = VectorStoreManager(settings)
    if not manager.list_documents():
        raise SystemExit("No indexed documents. Run: python ingest.py")
    try:
        service = RAGService(settings, Retriever(manager))
    except DocWiseError as exc:
        raise SystemExit(str(exc)) from exc
    history = ConversationHistory()
    options = RetrievalOptions(
        strategy=args.strategy or settings.retrieval_strategy,
        top_k=args.top_k or settings.top_k,
        rerank=not args.no_rerank and settings.rerank,
        max_context_characters=settings.max_context_characters,
        max_context_tokens=settings.max_context_tokens,
        advanced=settings.advanced_rag and not args.basic_rag,
        decompose=settings.query_decomposition,
        adaptive=settings.adaptive_retrieval,
        self_correct=settings.self_correction,
        verify_answer=settings.answer_verification,
        max_subqueries=settings.max_subqueries,
        max_adaptive_top_k=settings.max_adaptive_top_k,
        max_retrieval_attempts=settings.max_retrieval_attempts,
    )

    print(f"\nDocWise AI is ready ({options.strategy}, top-k={options.top_k}).")
    print("Type 'clear' to reset history or 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if question.lower() == "clear":
            history.clear()
            print("Conversation cleared.")
            continue
        if not question:
            continue
        try:
            response = service.answer(question, history=history, options=options)
        except (DocWiseError, ValueError) as exc:
            print(f"Error: {exc}")
            continue
        print(f"\nDocWise: {response.answer}")
        if response.model_name:
            print(f"Generated using: {response.model_name}")
        if response.fallback_reason:
            print(f"Model fallback: {response.fallback_reason}")
        if response.sources:
            print("\nSources:")
            for index, source in enumerate(response.sources, start=1):
                chunk = source.document.metadata.get("chunk_index", 0)
                print(
                    f"[{index}] {source.citation}, chunk {chunk} "
                    f"(relevance {source.score:.2f}; {source.retrieval_reason})"
                )
        if response.diagnostics:
            diagnostic = response.diagnostics
            print(
                f"\nRetrieval diagnostics: selected {diagnostic.selected_count}/"
                f"{diagnostic.total_candidates} chunks; per document "
                f"{diagnostic.per_document}"
            )
            if diagnostic.truncated:
                print(f"Warning: omitted chunks {diagnostic.omitted_chunks}")
            print(
                f"Complexity: {diagnostic.complexity}; attempts: "
                f"{diagnostic.retrieval_attempts}; Gemini calls: {diagnostic.llm_calls}"
            )
        history.add_user(question)
        history.add_assistant(
            response.answer,
            resolved_question=response.rewritten_question or question,
        )


if __name__ == "__main__":
    main()
