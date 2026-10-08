"""Automatic retrieval metrics and optional model-graded answer evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .comparison import is_comparison_request
from .intelligence import assess_evidence, plan_query, reformulate_query, verify_answer
from .models import RetrievedChunk
from .model_router import RoutingPreferences, TaskKind
from .retrieval import RetrievalOptions, Retriever, merge_retrieval_results


@dataclass(slots=True)
class EvaluationExample:
    question: str
    expected_sources: list[str]
    id: str = ""
    category: str = "factual"
    expected_behavior: str = "answer"
    expected_evidence: list[dict[str, Any]] | None = None
    expected_answer_contains: list[str] | None = None
    conversation: list[dict[str, str]] | None = None
    standalone_question: str | None = None


@dataclass(slots=True)
class EvaluationReport:
    examples: int
    retrieval_examples: int
    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float
    average_latency_ms: float
    average_retrieval_attempts: float
    average_llm_calls: float
    per_question: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_dataset(path: Path) -> list[EvaluationExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [EvaluationExample(**item) for item in payload]


def evaluate_retrieval(
    retriever: Retriever,
    examples: list[EvaluationExample],
    options: RetrievalOptions,
) -> EvaluationReport:
    """Calculate retrieval-only metrics without invoking an answer model."""
    details: list[dict[str, Any]] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    latencies: list[float] = []
    for example in examples:
        started = perf_counter()
        retrieval_question = example.standalone_question or example.question
        results = retriever.retrieve(retrieval_question, options)
        latency_ms = (perf_counter() - started) * 1000
        detail, metrics = _score_results(example, results)
        detail["latency_ms"] = latency_ms
        details.append(detail)
        latencies.append(latency_ms)
        if metrics is not None:
            recall, precision, reciprocal_rank = metrics
            recalls.append(recall)
            precisions.append(precision)
            reciprocal_ranks.append(reciprocal_rank)
    count = len(recalls)
    return EvaluationReport(
        examples=len(examples),
        retrieval_examples=count,
        recall_at_k=sum(recalls) / count if count else 0.0,
        precision_at_k=sum(precisions) / count if count else 0.0,
        mean_reciprocal_rank=sum(reciprocal_ranks) / count if count else 0.0,
        average_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        average_retrieval_attempts=1.0 if examples else 0.0,
        average_llm_calls=0.0,
        per_question=details,
    )


def compare_strategies(
    retriever: Retriever,
    examples: list[EvaluationExample],
    top_k: int = 4,
) -> dict[str, dict[str, Any]]:
    """Compare the four fixed retrieval strategies using the same labels."""
    reports = {}
    for name, strategy, rerank in (
        ("baseline_vector", "vector", False),
        ("bm25", "keyword", False),
        ("hybrid", "hybrid", False),
        ("hybrid_reranked", "hybrid", True),
    ):
        report = evaluate_retrieval(
            retriever,
            examples,
            RetrievalOptions(strategy=strategy, top_k=top_k, rerank=rerank),
        )
        reports[name] = report.to_dict()
    return reports


def evaluate_improved_retrieval(
    retriever: Retriever,
    examples: list[EvaluationExample],
    options: RetrievalOptions,
) -> EvaluationReport:
    """Evaluate adaptive, decomposed retrieval with the same objective labels."""
    details: list[dict[str, Any]] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    latencies: list[float] = []
    attempts: list[int] = []
    llm_calls: list[int] = []
    for example in examples:
        retrieval_question = example.standalone_question or example.question
        comparison = is_comparison_request(retrieval_question)
        plan = plan_query(
            retrieval_question,
            comparison=comparison,
            base_top_k=options.top_k,
            max_top_k=options.max_adaptive_top_k,
            max_subqueries=options.max_subqueries,
            decompose=options.decompose,
            adaptive=options.adaptive,
        )
        started = perf_counter()
        if hasattr(retriever, "retrieve_intelligently"):
            results, diagnostics = retriever.retrieve_intelligently(
                retrieval_question,
                list(plan.subqueries),
                options,
                comparison_mode=comparison,
                complexity=plan.complexity,
                adaptive_top_k=plan.adaptive_top_k,
            )
            assessment = assess_evidence(
                plan,
                results,
                minimum_score=0.15,
                subquery_matches=diagnostics.subquery_matches,
                comparison=comparison,
            )
            if (
                not assessment.sufficient
                and options.self_correct
                and options.max_retrieval_attempts > 1
            ):
                retry_query = reformulate_query(plan, assessment)
                retry_results, retry_diagnostics = retriever.retrieve_intelligently(
                    retry_query,
                    [retry_query],
                    options,
                    comparison_mode=comparison,
                    complexity=plan.complexity,
                    adaptive_top_k=options.max_adaptive_top_k,
                )
                results, _ = merge_retrieval_results(
                    results,
                    retry_results,
                    options,
                    comparison_mode=comparison,
                )
                diagnostics.retrieval_attempts = 2
                diagnostics.additional_retrieval_attempts = 1
                diagnostics.reformulated_query = retry_query
                diagnostics.subquery_matches.update(retry_diagnostics.subquery_matches)
        else:
            results = retriever.retrieve(retrieval_question, options)
            diagnostics = None
        latency_ms = (perf_counter() - started) * 1000
        detail, metrics = _score_results(example, results)
        detail.update(
            {
                "latency_ms": latency_ms,
                "subqueries": list(plan.subqueries),
                "adaptive_top_k": plan.adaptive_top_k,
                "retrieval_attempts": diagnostics.retrieval_attempts if diagnostics else 1,
                "llm_calls": diagnostics.llm_calls if diagnostics else 0,
            }
        )
        details.append(detail)
        latencies.append(latency_ms)
        attempts.append(detail["retrieval_attempts"])
        llm_calls.append(detail["llm_calls"])
        if metrics is not None:
            recall, precision, reciprocal_rank = metrics
            recalls.append(recall)
            precisions.append(precision)
            reciprocal_ranks.append(reciprocal_rank)
    count = len(recalls)
    return EvaluationReport(
        examples=len(examples),
        retrieval_examples=count,
        recall_at_k=sum(recalls) / count if count else 0.0,
        precision_at_k=sum(precisions) / count if count else 0.0,
        mean_reciprocal_rank=sum(reciprocal_ranks) / count if count else 0.0,
        average_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        average_retrieval_attempts=sum(attempts) / len(attempts) if attempts else 0.0,
        average_llm_calls=sum(llm_calls) / len(llm_calls) if llm_calls else 0.0,
        per_question=details,
    )


def compare_baseline_improved(
    retriever: Retriever,
    examples: list[EvaluationExample],
    top_k: int = 4,
) -> dict[str, dict[str, Any]]:
    """Compare fixed-depth vector RAG with the advanced hybrid retrieval pipeline."""
    baseline = evaluate_retrieval(
        retriever,
        examples,
        RetrievalOptions(strategy="vector", top_k=top_k, rerank=False, advanced=False),
    )
    improved = evaluate_improved_retrieval(
        retriever,
        examples,
        RetrievalOptions(strategy="hybrid", top_k=top_k, rerank=True, advanced=True),
    )
    return {"baseline_rag": baseline.to_dict(), "improved_rag": improved.to_dict()}


def benchmark_retrieval_strategies(
    retriever: Retriever,
    examples: list[EvaluationExample],
    top_k: int = 4,
) -> dict[str, dict[str, Any]]:
    """Run all requested retrieval variants without any Gemini calls."""
    reports = compare_strategies(retriever, examples, top_k=top_k)
    improved = evaluate_improved_retrieval(
        retriever,
        examples,
        RetrievalOptions(strategy="hybrid", top_k=top_k, rerank=True, advanced=True),
    )
    reports["intelligent_self_correcting"] = improved.to_dict()
    return reports


def evaluate_answer_outputs(
    examples: list[EvaluationExample],
    outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Measure answer-level quality separately from retrieval-only benchmarking."""
    citation_scores: list[float] = []
    support_scores: list[float] = []
    refusal_scores: list[float] = []
    attempts: list[int] = []
    llm_calls: list[int] = []
    details: list[dict[str, Any]] = []
    for example in examples:
        key = example.id or example.question
        output = outputs.get(key)
        if output is None:
            continue
        answer = str(output.get("answer", ""))
        sources: list[RetrievedChunk] = list(output.get("sources", []))
        labels = [
            int(value)
            for group in re.findall(r"\[([\d,\s]+)\]", answer)
            for value in re.findall(r"\d+", group)
        ]
        valid = sum(1 for label in labels if 1 <= label <= len(sources))
        is_refusal = answer.startswith("I couldn't find that information")
        citation_validity = (
            valid / len(labels)
            if labels
            else (1.0 if not answer or is_refusal else 0.0)
        )
        verification = verify_answer(answer, sources)
        support_accuracy = verification.citation_correctness
        refusal_correct = (
            float(is_refusal)
            if example.expected_behavior == "refuse"
            else float(not is_refusal)
        )
        diagnostics = output.get("diagnostics")
        retrieval_attempts = int(getattr(diagnostics, "retrieval_attempts", 1))
        model_calls = int(getattr(diagnostics, "llm_calls", 0))
        citation_scores.append(citation_validity)
        if support_accuracy is not None:
            support_scores.append(support_accuracy)
        refusal_scores.append(refusal_correct)
        attempts.append(retrieval_attempts)
        llm_calls.append(model_calls)
        details.append(
            {
                "id": example.id,
                "question": example.question,
                "citation_validity": citation_validity,
                "evidence_support_accuracy": support_accuracy,
                "refusal_correct": bool(refusal_correct),
                "retrieval_attempts": retrieval_attempts,
                "llm_calls": model_calls,
            }
        )
    count = len(details)
    return {
        "examples": count,
        "citation_validity": sum(citation_scores) / count if count else 0.0,
        "evidence_support_accuracy": (
            sum(support_scores) / len(support_scores) if support_scores else 0.0
        ),
        "unsupported_answer_refusal_accuracy": (
            sum(refusal_scores) / count if count else 0.0
        ),
        "average_retrieval_attempts": sum(attempts) / count if count else 0.0,
        "average_llm_calls": sum(llm_calls) / count if count else 0.0,
        "per_question": details,
    }


def _score_results(
    example: EvaluationExample,
    results: list[RetrievedChunk],
) -> tuple[dict[str, Any], tuple[float, float, float] | None]:
    actual = [str(item.document.metadata.get("source", "")) for item in results]
    detail = {
        "id": example.id,
        "category": example.category,
        "question": example.question,
        "standalone_question": example.standalone_question,
        "expected_sources": example.expected_sources,
        "expected_evidence": example.expected_evidence or [],
        "expected_behavior": example.expected_behavior,
        "retrieved_sources": actual,
    }
    if not example.expected_sources and not example.expected_evidence:
        detail.update({"recall_at_k": None, "precision_at_k": None, "reciprocal_rank": None})
        return detail, None

    if example.expected_evidence:
        units = example.expected_evidence
        matched_units = {
            index
            for index, unit in enumerate(units)
            if any(_matches_evidence_unit(item, unit) for item in results)
        }
        relevant_flags = [
            any(_matches_evidence_unit(item, unit) for unit in units) for item in results
        ]
        recall = len(matched_units) / len(units)
    else:
        expected = set(example.expected_sources)
        recall = len(expected.intersection(actual)) / len(expected)
        relevant_flags = [source in expected for source in actual]
    precision = sum(relevant_flags) / len(results) if results else 0.0
    first_rank = next(
        (index for index, relevant in enumerate(relevant_flags, start=1) if relevant),
        None,
    )
    reciprocal_rank = 1 / first_rank if first_rank else 0.0
    detail.update(
        {
            "recall_at_k": recall,
            "precision_at_k": precision,
            "reciprocal_rank": reciprocal_rank,
        }
    )
    return detail, (recall, precision, reciprocal_rank)


def _matches_evidence_unit(item: RetrievedChunk, unit: dict[str, Any]) -> bool:
    source = str(item.document.metadata.get("source", ""))
    if source != str(unit.get("source", "")):
        return False
    expected_text = unit.get("contains", [])
    terms = [expected_text] if isinstance(expected_text, str) else list(expected_text)
    content = item.document.page_content.lower()
    return all(str(term).lower() in content for term in terms)


def llm_grade_answer(
    question: str,
    answer: str,
    evidence: str,
    judge: Any,
    routing_preferences: RoutingPreferences | None = None,
) -> dict[str, Any]:
    """Optional subjective judge; kept separate from deterministic retrieval metrics."""
    prompt = (
        "Grade the answer against the evidence on groundedness and completeness from 1 to 5. "
        "Return JSON with groundedness, completeness, and rationale.\n"
        f"Question: {question}\nAnswer: {answer}\nEvidence: {evidence}"
    )
    if hasattr(judge, "invoke_routed"):
        response = judge.invoke_routed(
            prompt,
            purpose="evaluation_grading",
            task=TaskKind.EVALUATION,
            preferences=routing_preferences or RoutingPreferences(),
        ).message
    else:
        response = judge.invoke(prompt)
    text = getattr(response, "content", response)
    try:
        return json.loads(str(text))
    except json.JSONDecodeError:
        return {"raw_model_assessment": str(text), "note": "Subjective model-graded result."}
