from langchain_core.documents import Document

from pathlib import Path

from docwise.evaluation import (
    EvaluationExample,
    benchmark_retrieval_strategies,
    evaluate_answer_outputs,
    evaluate_retrieval,
    llm_grade_answer,
    load_dataset,
)
from docwise.model_router import RoutedInvocation
from docwise.models import RetrievedChunk
from docwise.retrieval import RetrievalOptions


class StaticRetriever:
    def retrieve(self, question, options):
        return [
            RetrievedChunk(Document(page_content="x", metadata={"source": "wrong.pdf"}), 0.8),
            RetrievedChunk(Document(page_content="y", metadata={"source": "right.pdf"}), 0.7),
        ]


def test_recall_and_mrr_are_calculated_separately():
    report = evaluate_retrieval(
        StaticRetriever(),
        [EvaluationExample("question", ["right.pdf"])],
        RetrievalOptions(top_k=2),
    )
    assert report.recall_at_k == 1.0
    assert report.mean_reciprocal_rank == 0.5


def test_llm_grading_uses_tracked_router_path_when_available():
    class GradeMessage:
        content = '{"groundedness": 5, "completeness": 4, "rationale": "Supported"}'

    class GradeRouter:
        def __init__(self):
            self.kwargs = None

        def invoke_routed(self, prompt, **kwargs):
            self.kwargs = kwargs
            return RoutedInvocation(GradeMessage(), "gemini-3.5-flash-lite", None, (), 10, 4)

    router = GradeRouter()
    result = llm_grade_answer("Question", "Answer", "Evidence", router)
    assert result["groundedness"] == 5
    assert router.kwargs["purpose"] == "evaluation_grading"


def test_benchmark_dataset_has_defensible_manual_labels_and_at_least_40_examples():
    root = Path(__file__).resolve().parents[1]
    examples = load_dataset(root / "evaluation" / "docwise_benchmark_40.json")

    assert len(examples) >= 40
    assert len({example.id for example in examples}) == len(examples)
    for example in examples:
        if example.expected_behavior == "refuse":
            assert example.expected_sources == []
            continue
        assert example.expected_sources
        assert example.expected_evidence
        assert all(
            unit["source"] in example.expected_sources
            for unit in example.expected_evidence
        )


class FiveStrategyRetriever:
    def retrieve(self, question, options):
        return [
            RetrievedChunk(
                Document(page_content="target evidence", metadata={"source": "right.pdf"}),
                0.9,
            )
        ]

    def retrieve_intelligently(self, question, subqueries, options, **kwargs):
        chunks = self.retrieve(question, options)
        from docwise.models import RetrievalDiagnostics

        return chunks, RetrievalDiagnostics(
            query=question,
            strategy=options.strategy,
            comparison_mode=False,
            total_candidates=1,
            selected_count=1,
            subqueries=subqueries,
            subquery_matches={query: 1 for query in subqueries},
            adaptive_top_k=options.top_k,
        )


def test_five_strategy_benchmark_separates_retrieval_metrics_from_llm_calls():
    reports = benchmark_retrieval_strategies(
        FiveStrategyRetriever(),
        [EvaluationExample("question", ["right.pdf"])],
        top_k=1,
    )

    assert set(reports) == {
        "baseline_vector",
        "bm25",
        "hybrid",
        "hybrid_reranked",
        "intelligent_self_correcting",
    }
    assert all(report["recall_at_k"] == 1.0 for report in reports.values())
    assert all(report["precision_at_k"] == 1.0 for report in reports.values())
    assert all(report["average_llm_calls"] == 0.0 for report in reports.values())


def test_answer_quality_metrics_measure_citations_support_and_refusals_separately():
    supported = EvaluationExample(
        "How much leave?",
        ["policy.pdf"],
        id="supported",
    )
    unsupported = EvaluationExample(
        "Who is the CEO?",
        [],
        id="missing",
        expected_behavior="refuse",
    )
    source = RetrievedChunk(
        Document(
            page_content="Employees receive 20 days of annual leave.",
            metadata={"source": "policy.pdf"},
        ),
        0.9,
    )
    report = evaluate_answer_outputs(
        [supported, unsupported],
        {
            "supported": {"answer": "Employees receive 20 days [1].", "sources": [source]},
            "missing": {
                "answer": "I couldn't find that information in the indexed documents.",
                "sources": [],
            },
        },
    )

    assert report["citation_validity"] == 1.0
    assert report["evidence_support_accuracy"] == 1.0
    assert report["unsupported_answer_refusal_accuracy"] == 1.0
