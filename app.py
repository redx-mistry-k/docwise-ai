"""Streamlit web interface for DocWise AI."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from docwise.config import Settings
from docwise.document_loader import safe_filename
from docwise.exceptions import DocWiseError
from docwise.history import ConversationHistory
from docwise.model_router import (
    GeminiModelRouter,
    RoutingMode,
    TaskKind,
    routing_preferences_from_session_state,
)
from docwise.rag import RAGService
from docwise.retrieval import RetrievalOptions, Retriever
from docwise.security import (
    access_password,
    check_session_rate_limit,
    password_matches,
    query_limit_per_minute,
)
from docwise.vector_store import VectorStoreManager

st.set_page_config(page_title="DocWise AI", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 2rem;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
      .docwise-hero {padding: .3rem 0 1.2rem 0;}
      .docwise-hero h1 {margin-bottom: .2rem; letter-spacing: -.03em;}
      .docwise-muted {opacity: .72;}
    </style>
    """,
    unsafe_allow_html=True,
)

MAX_SESSION_MESSAGES = 40


def trim_session_messages() -> None:
    """Bound retained UI payloads while ConversationHistory keeps its own smaller window."""
    st.session_state.messages = st.session_state.messages[-MAX_SESSION_MESSAGES:]

required_password = access_password()
if required_password and not st.session_state.get("docwise_authenticated", False):
    st.title("DocWise AI")
    st.caption("Enter the deployment access password to continue.")
    submitted_password = st.text_input("Access password", type="password")
    if st.button("Sign in", type="primary"):
        if password_matches(submitted_password, required_password):
            st.session_state.docwise_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect access password.")
    st.stop()


@st.cache_resource(show_spinner="Loading DocWise resources…")
def resources() -> tuple[Settings, VectorStoreManager, GeminiModelRouter]:
    settings = Settings.from_env()
    return settings, VectorStoreManager(settings), GeminiModelRouter.from_settings(settings)


def history_from_session() -> ConversationHistory:
    return ConversationHistory.from_messages(st.session_state.messages)


def render_sources(sources: list[Any]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for index, source in enumerate(sources, start=1):
            metadata = source.document.metadata
            snippet = " ".join(source.document.page_content.split())
            st.markdown(
                f"**[{index}] {metadata.get('source', 'Unknown')} — page "
                f"{metadata.get('page', 1)}, chunk {metadata.get('chunk_index', 0)}** · "
                f"relevance `{source.score:.2f}` · {source.retrieval_reason}"
            )
            st.caption(snippet[:700] + ("…" if len(snippet) > 700 else ""))


def render_diagnostics(diagnostics: Any | None, sources: list[Any]) -> None:
    if diagnostics is None:
        return
    with st.expander("Retrieval diagnostics", expanded=False):
        mode = "Document comparison" if diagnostics.comparison_mode else "Standard RAG"
        st.caption(
            f"{mode} · {diagnostics.strategy} · selected {diagnostics.selected_count} of "
            f"{diagnostics.total_candidates} available chunks · "
            f"{diagnostics.context_characters:,} context characters · "
            f"approximately {diagnostics.context_tokens:,} tokens"
        )
        st.caption(
            f"Complexity: {diagnostics.complexity} · adaptive top-k: "
            f"{diagnostics.adaptive_top_k} · retrieval attempts: "
            f"{diagnostics.retrieval_attempts} · Gemini calls: {diagnostics.llm_calls}"
        )
        st.code(diagnostics.query, language=None)
        if diagnostics.subqueries:
            st.markdown("**Focused retrieval queries**")
            for subquery in diagnostics.subqueries:
                matches = diagnostics.subquery_matches.get(subquery, 0)
                st.markdown(f"- {subquery} — {matches} relevant match(es)")
        if diagnostics.reformulated_query:
            st.markdown("**Bounded retry query**")
            st.code(diagnostics.reformulated_query, language=None)
        for filename, count in diagnostics.per_document.items():
            st.markdown(f"- **{filename}:** {count} retrieved chunk(s)")
        for index, source in enumerate(sources, start=1):
            metadata = source.document.metadata
            st.caption(
                f"[{index}] {metadata.get('source', 'Unknown')} · page "
                f"{metadata.get('page', 1)} · chunk {metadata.get('chunk_index', 0)} · "
                f"{source.retrieval_reason} · score {source.score:.3f}"
            )
        if diagnostics.truncated:
            st.warning(
                f"Context budget omitted {len(diagnostics.omitted_chunks)} chunk(s): "
                + ", ".join(diagnostics.omitted_chunks)
            )
        if diagnostics.verification_claims:
            correctness = diagnostics.citation_correctness
            st.caption(
                f"Answer verification: {diagnostics.supported_claims}/"
                f"{diagnostics.verification_claims} checked claims supported · "
                f"citation correctness {correctness:.0%}"
                if correctness is not None
                else "Answer verification: no factual claims required checking."
            )
        if diagnostics.invalid_citations:
            st.warning(
                "Removed invalid citation labels: "
                + ", ".join(map(str, diagnostics.invalid_citations))
            )
        if diagnostics.unsupported_claims:
            st.warning(
                f"Removed {len(diagnostics.unsupported_claims)} unsupported claim(s)."
            )
        if diagnostics.uncertain_claims:
            st.info(
                f"Preserved {len(diagnostics.uncertain_claims)} cited paraphrase(s) that the "
                "local lexical verifier could not conclusively validate."
            )
        if diagnostics.contradictory_claims:
            st.warning(
                f"Removed {len(diagnostics.contradictory_claims)} contradictory claim(s)."
            )
        for conflict in diagnostics.evidence_contradictions:
            st.warning(conflict)
        for warning in diagnostics.warnings:
            st.warning(warning)


def render_model_metadata(model_name: str | None, fallback_reason: str | None) -> None:
    if model_name:
        st.caption(f"Generated using: `{model_name}`")
    if fallback_reason:
        st.info(f"Model fallback: {fallback_reason}")


try:
    settings, manager, model_router = resources()
except Exception as exc:
    st.error(f"DocWise could not start: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("📚 DocWise AI")
    st.caption("Intelligent Document RAG System")
    uploads = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        help=f"PDF, TXT, or Markdown; up to {settings.max_file_size_mb} MB each.",
    )
    upload_chunk_size = st.number_input(
        "Chunk size", min_value=200, max_value=4000, value=settings.chunk_size, step=100
    )
    upload_overlap = st.number_input(
        "Chunk overlap",
        min_value=0,
        max_value=max(0, int(upload_chunk_size) - 1),
        value=min(settings.chunk_overlap, int(upload_chunk_size) - 1),
        step=25,
    )
    if st.button("Index uploaded documents", type="primary", use_container_width=True):
        if not uploads:
            st.warning("Choose at least one document first.")
        else:
            for upload in uploads:
                try:
                    if upload.size > settings.max_file_size_mb * 1024 * 1024:
                        raise ValueError(
                            f"{upload.name} exceeds the {settings.max_file_size_mb} MB limit."
                        )
                    with tempfile.TemporaryDirectory(dir=settings.storage_dir) as temporary:
                        staged = Path(temporary) / safe_filename(upload.name)
                        staged.write_bytes(upload.getvalue())
                        result = manager.index_file(
                            staged,
                            chunk_size=int(upload_chunk_size),
                            chunk_overlap=int(upload_overlap),
                            filename_override=upload.name,
                        )
                    (st.success if result.status == "indexed" else st.info)(result.message)
                except Exception as exc:
                    st.error(f"{upload.name}: {exc}")

    st.divider()
    st.subheader("Retrieval")
    valid_strategy = (
        settings.retrieval_strategy
        if settings.retrieval_strategy in {"hybrid", "vector", "keyword"}
        else "hybrid"
    )
    strategy = st.selectbox(
        "Method",
        ["hybrid", "vector", "keyword"],
        index=["hybrid", "vector", "keyword"].index(valid_strategy),
        format_func=lambda value: {
            "hybrid": "Hybrid (recommended)",
            "vector": "Semantic vector",
            "keyword": "Keyword BM25",
        }[value],
    )
    top_k = st.slider("Retrieved chunks", 1, 12, settings.top_k)
    rerank = st.toggle("Local reranking", value=settings.rerank)
    advanced_rag = st.toggle("Advanced accuracy checks", value=settings.advanced_rag)
    with st.expander("Advanced retrieval settings"):
        query_decomposition = st.toggle(
            "Decompose multi-part questions",
            value=settings.query_decomposition,
            disabled=not advanced_rag,
        )
        adaptive_retrieval = st.toggle(
            "Adaptive retrieval depth",
            value=settings.adaptive_retrieval,
            disabled=not advanced_rag,
        )
        self_correction = st.toggle(
            "One bounded retrieval retry",
            value=settings.self_correction,
            disabled=not advanced_rag,
        )
        answer_verification = st.toggle(
            "Verify answer claims and citations",
            value=settings.answer_verification,
            disabled=not advanced_rag,
        )
    st.caption(
        "Complex questions can expand retrieval locally. Advanced checks do not add Gemini "
        "calls by default."
    )

    st.divider()
    st.subheader("Model Management")
    mode_labels = {
        RoutingMode.BALANCED.value: "Balanced",
        RoutingMode.QUALITY_FIRST.value: "Quality First",
        RoutingMode.QUOTA_SAVER.value: "Quota Saver",
    }
    configured_mode = (
        settings.routing_mode
        if settings.routing_mode in mode_labels
        else RoutingMode.BALANCED.value
    )
    routing_mode_value = st.selectbox(
        "Routing mode",
        list(mode_labels),
        index=list(mode_labels).index(configured_mode),
        format_func=lambda value: mode_labels[value],
        key="gemini_routing_mode",
    )
    selection_options = ["automatic", "lite_only", *model_router.config.priority]
    manual_selection_value = st.selectbox(
        "Model restriction",
        selection_options,
        format_func=lambda value: {
            "automatic": "Automatic",
            "lite_only": "Lite models only",
        }.get(value, value),
        key="gemini_manual_selection",
    )
    priority_text = st.text_area(
        "Model priority (highest first)",
        value="\n".join(model_router.config.priority),
        height=150,
        key="gemini_priority_text",
        help="Use each approved model exactly once. Paid or unknown models are rejected.",
    )
    requested_priority = tuple(
        value.strip() for value in priority_text.replace(",", "\n").splitlines() if value.strip()
    )
    try:
        priority_override = model_router.config.validate_priority(requested_priority)
    except ValueError as exc:
        st.error(str(exc))
        priority_override = model_router.config.priority
    routing_preferences = routing_preferences_from_session_state(
        {
            "gemini_routing_mode": routing_mode_value,
            "gemini_manual_selection": manual_selection_value,
            "gemini_priority": priority_override,
        },
        model_router.config,
    )
    active_model = model_router.select_model(TaskKind.SIMPLE, routing_preferences)
    last_model = model_router.tracker.get_state("last_model_used")
    st.caption(f"Current active model: `{active_model or 'none available'}`")
    if last_model:
        st.caption(f"Last model used: `{last_model}`")
    last_fallback = model_router.tracker.get_state("last_fallback_reason")
    if last_fallback:
        st.caption(f"Last fallback: {last_fallback}")
    reset = model_router.tracker.next_reset()
    st.caption(f"Next daily reset: {reset.strftime('%Y-%m-%d %I:%M %p %Z')}")
    st.caption("Counts are local estimates; other apps using the same Google project are not visible.")
    credential_source = getattr(model_router, "credential_source", "GOOGLE_API_KEY")
    st.caption(f"Gemini credential source: `{credential_source}` (value hidden)")
    if getattr(model_router, "multiple_credentials_configured", False):
        st.info(
            "Both GOOGLE_API_KEY and GEMINI_API_KEY are configured. DocWise explicitly uses "
            "GOOGLE_API_KEY; neither environment variable was changed."
        )
    with st.expander("Free-tier usage estimates"):
        for model_status in model_router.statuses():
            status_label = model_status.reason or "available"
            st.markdown(
                f"**{model_status.model_id}** — attempted `{model_status.attempted_today}`, "
                f"successful `{model_status.successful_today}`, "
                f"remaining `{model_status.estimated_remaining}` · {status_label}"
            )

    st.divider()
    records = manager.list_documents()
    st.subheader(f"Indexed documents ({len(records)})")
    if not records:
        st.caption("No documents indexed yet.")
    for record in records:
        with st.expander(record.filename):
            st.caption(f"{record.page_count} page(s) · {record.chunk_count} chunks")
            left, right = st.columns(2)
            if left.button("Reindex", key=f"reindex-{record.document_id}", use_container_width=True):
                try:
                    manager.reindex_document(
                        record.document_id,
                        chunk_size=int(upload_chunk_size),
                        chunk_overlap=int(upload_overlap),
                    )
                    st.success("Reindexed.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            if right.button("Delete", key=f"delete-{record.document_id}", use_container_width=True):
                try:
                    manager.delete_document(record.document_id)
                    st.session_state.messages = []
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown(
    '<div class="docwise-hero"><h1>Chat with your documents</h1>'
    '<div class="docwise-muted">Grounded answers with page-level evidence.</div></div>',
    unsafe_allow_html=True,
)
stats = manager.stats()
metric_columns = st.columns(4)
metric_columns[0].metric("Documents", stats["document_count"])
metric_columns[1].metric("Chunks", stats["chunk_count"])
metric_columns[2].metric("Embeddings", settings.embedding_model.split("/")[-1])
metric_columns[3].metric("LLM", active_model or "No free model")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            render_diagnostics(message.get("diagnostics"), message.get("sources", []))
            render_model_metadata(
                message.get("model_name"), message.get("fallback_reason")
            )

question = st.chat_input("Ask a question about your indexed documents…")
if question:
    rate_decision = check_session_rate_limit(
        st.session_state.get("query_timestamps", []),
        limit=query_limit_per_minute(),
    )
    st.session_state.query_timestamps = list(rate_decision.timestamps)
    if not rate_decision.allowed:
        st.error(
            "Too many questions in this session. Try again in "
            f"approximately {rate_decision.retry_after_seconds} seconds."
        )
        st.stop()
    prior_history = history_from_session()
    resolved_question = question
    diagnostics = None
    model_name = None
    fallback_reason = None
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not manager.list_documents():
            answer = "Upload and index at least one document before asking a question."
            st.warning(answer)
            sources = []
        else:
            try:
                service = RAGService(
                    settings,
                    Retriever(manager),
                    llm=model_router,
                    routing_preferences=routing_preferences,
                )
                options = RetrievalOptions(
                    strategy=strategy,
                    top_k=top_k,
                    rerank=rerank,
                    max_context_characters=settings.max_context_characters,
                    max_context_tokens=settings.max_context_tokens,
                    advanced=advanced_rag,
                    decompose=query_decomposition,
                    adaptive=adaptive_retrieval,
                    self_correct=self_correction,
                    verify_answer=answer_verification,
                    max_subqueries=settings.max_subqueries,
                    max_adaptive_top_k=settings.max_adaptive_top_k,
                    max_retrieval_attempts=settings.max_retrieval_attempts,
                )
                with st.spinner("Searching documents and grounding the answer…"):
                    response = service.answer(question, history=prior_history, options=options)
                answer, sources = response.answer, response.sources
                resolved_question = response.rewritten_question or question
                diagnostics = response.diagnostics
                model_name = response.model_name
                fallback_reason = response.fallback_reason
                st.markdown(answer)
                render_sources(sources)
                render_diagnostics(diagnostics, sources)
                render_model_metadata(model_name, fallback_reason)
            except (DocWiseError, ValueError) as exc:
                answer, sources = f"I couldn't complete that request: {exc}", []
                st.error(answer)
            except Exception as exc:
                answer, sources = f"Unexpected application error: {exc}", []
                st.error(answer)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "diagnostics": diagnostics,
                "model_name": model_name,
                "fallback_reason": fallback_reason,
                "resolved_question": resolved_question,
            }
        )
        trim_session_messages()
