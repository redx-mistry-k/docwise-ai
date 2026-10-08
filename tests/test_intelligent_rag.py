from langchain_core.documents import Document

from docwise.evaluation import EvaluationExample, compare_baseline_improved
from docwise.intelligence import plan_query, verify_answer
from docwise.models import RetrievalDiagnostics, RetrievedChunk
from docwise.rag import INSUFFICIENT_ANSWER, RAGService
from docwise.retrieval import RetrievalOptions, Retriever


def retrieved(text, source="policy.pdf", chunk_id="policy:0", score=0.9):
    return RetrievedChunk(
        Document(
            page_content=text,
            metadata={
                "source": source,
                "page": 1,
                "chunk_index": 0,
                "chunk_id": chunk_id,
            },
        ),
        score=score,
    )


def diagnostics(query, chunks, matches):
    return RetrievalDiagnostics(
        query=query,
        strategy="hybrid",
        comparison_mode=False,
        total_candidates=len(chunks),
        selected_count=len(chunks),
        per_document={chunk.document.metadata["source"]: 1 for chunk in chunks},
        context_characters=sum(len(chunk.document.page_content) for chunk in chunks),
        subqueries=[query],
        subquery_matches={query: matches},
        adaptive_top_k=4,
    )


class OneRetryRetriever:
    def __init__(self, recover=True):
        self.recover = recover
        self.calls = []

    def retrieve_intelligently(
        self,
        query,
        subqueries,
        options,
        *,
        comparison_mode,
        complexity,
        adaptive_top_k,
    ):
        self.calls.append(query)
        if len(self.calls) == 1:
            chunks = [retrieved("Unrelated cafeteria menu.", "other.txt", "other:0", 0.02)]
            return chunks, diagnostics(query, chunks, 0)
        score = 0.91 if self.recover else 0.02
        text = "The policy grants 20 days of annual leave." if self.recover else "Still unrelated."
        chunks = [retrieved(text, score=score)]
        return chunks, diagnostics(query, chunks, 1 if self.recover else 0)


class StaticLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.response


def test_query_planner_decomposes_multi_requirement_but_not_simple_question():
    complex_plan = plan_query(
        "Explain the inventory project. What technologies were used and what business "
        "problem did it solve?",
        comparison=False,
        base_top_k=4,
        max_top_k=10,
        max_subqueries=4,
        decompose=True,
        adaptive=True,
    )
    simple_plan = plan_query(
        "What is the annual leave allowance?",
        comparison=False,
        base_top_k=4,
        max_top_k=10,
        max_subqueries=4,
        decompose=True,
        adaptive=True,
    )

    assert complex_plan.decomposed is True
    assert len(complex_plan.subqueries) >= 3
    assert complex_plan.adaptive_top_k > 4
    assert simple_plan.subqueries == (simple_plan.original_query,)
    assert simple_plan.adaptive_top_k == 4


def test_self_correcting_rag_performs_one_local_retry_and_no_extra_llm_call(settings):
    retriever = OneRetryRetriever(recover=True)
    llm = StaticLLM("The policy grants 20 days of annual leave [1].")
    response = RAGService(settings, retriever, llm).answer(
        "What is the leave allowance?",
        options=RetrievalOptions(strategy="hybrid", top_k=4),
    )

    assert response.answer == "The policy grants 20 days of annual leave [1]."
    assert len(retriever.calls) == 2
    assert len(llm.calls) == 1
    assert response.diagnostics.retrieval_attempts == 2
    assert response.diagnostics.additional_retrieval_attempts == 1
    assert response.diagnostics.llm_calls == 1
    assert response.diagnostics.reformulated_query


def test_self_correction_is_bounded_when_evidence_remains_irrelevant(settings):
    retriever = OneRetryRetriever(recover=False)
    llm = StaticLLM("This must not be called.")
    response = RAGService(settings, retriever, llm).answer(
        "What is the leave allowance?",
        options=RetrievalOptions(strategy="hybrid", top_k=4),
    )

    assert response.answer == INSUFFICIENT_ANSWER
    assert len(retriever.calls) == 2
    assert llm.calls == []
    assert response.diagnostics.retrieval_attempts == 2


def test_verifier_rejects_unrelated_invalid_and_contradictory_claims():
    sources = [retrieved("The policy provides 20 days of annual leave.")]
    report = verify_answer(
        "Annual leave is 20 days [1].\n"
        "Annual leave is 99 days [1].\n"
        "The policy does not provide annual leave [1].\n"
        "An invented claim [9].",
        sources,
    )

    assert "Annual leave is 20 days [1]." in report.answer
    assert "99 days" not in report.answer
    assert "does not provide" not in report.answer
    assert report.supported_claims == 1
    assert len(report.unsupported_claims) == 2
    assert len(report.contradictory_claims) == 1
    assert report.invalid_citations == [9]
    assert report.citation_correctness == 0.25


def test_verifier_preserves_uncertain_paraphrase_instead_of_false_rejection():
    sources = [
        retrieved("Personnel may perform their duties away from company premises.")
    ]
    report = verify_answer("Employees are allowed to work remotely [1].", sources)

    assert report.answer == "Employees are allowed to work remotely [1]."
    assert report.supported_claims == 0
    assert report.uncertain_claims == ["Employees are allowed to work remotely ."]


def test_verifier_flags_conflicting_retrieved_sources():
    sources = [
        retrieved("Remote work is allowed two days per week.", "policy-a.md", "a:0"),
        retrieved("Remote work is not allowed two days per week.", "policy-b.md", "b:0"),
    ]
    report = verify_answer("Remote work is allowed two days per week [1].", sources)

    assert report.evidence_contradictions
    assert "policy-a.md" in report.evidence_contradictions[0]
    assert "policy-b.md" in report.evidence_contradictions[0]


class RawChunkStore:
    def __init__(self, chunks):
        self._chunks = chunks
        self.all_chunk_calls = 0

    def all_chunks(self):
        self.all_chunk_calls += 1
        return self._chunks

    def list_documents(self):
        return []


def raw(source, index, text):
    return {
        "id": f"{source}:{index}",
        "text": text,
        "metadata": {
            "source": source,
            "page": 1,
            "chunk_index": index,
            "chunk_id": f"{source}:{index}",
        },
    }


def test_intelligent_retrieval_deduplicates_and_respects_token_budget():
    store = RawChunkStore(
        [
            raw("resume.pdf", 0, "Power Apps SharePoint inventory technology tools"),
            raw("resume.pdf", 1, "Inventory stock tracking business problem " * 30),
            raw("notes.txt", 0, "Unrelated notes"),
        ]
    )
    retriever = Retriever(store)
    question = "What technologies did the inventory project use and what problem did it solve?"
    plan = plan_query(
        question,
        comparison=False,
        base_top_k=2,
        max_top_k=6,
        max_subqueries=4,
        decompose=True,
        adaptive=True,
    )
    results, report = retriever.retrieve_intelligently(
        question,
        list(plan.subqueries),
        RetrievalOptions(
            strategy="keyword",
            top_k=2,
            max_adaptive_top_k=6,
            max_context_tokens=20,
        ),
        complexity=plan.complexity,
        adaptive_top_k=plan.adaptive_top_k,
    )

    ids = [item.document.metadata["chunk_id"] for item in results]
    assert len(ids) == len(set(ids))
    assert report.subqueries == list(plan.subqueries)
    assert report.adaptive_top_k > 2
    assert report.context_tokens <= 20
    assert report.truncated is True
    assert store.all_chunk_calls == 1


class BaselineImprovedRetriever:
    def retrieve(self, question, options):
        return [retrieved("irrelevant", "wrong.pdf", "wrong:0")]

    def retrieve_intelligently(self, question, subqueries, options, **kwargs):
        chunks = [retrieved("relevant", "right.pdf", "right:0")]
        return chunks, diagnostics(question, chunks, 1)


def test_baseline_vs_improved_evaluation_reports_recall_and_mrr_gain():
    report = compare_baseline_improved(
        BaselineImprovedRetriever(),
        [EvaluationExample("multi-part question", ["right.pdf"])],
        top_k=1,
    )

    assert report["baseline_rag"]["recall_at_k"] == 0.0
    assert report["baseline_rag"]["mean_reciprocal_rank"] == 0.0
    assert report["improved_rag"]["recall_at_k"] == 1.0
    assert report["improved_rag"]["mean_reciprocal_rank"] == 1.0
