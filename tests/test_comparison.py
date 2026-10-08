from datetime import date

from langchain_core.documents import Document

from docwise.comparison import (
    expand_citations,
    extract_employment_history,
    filter_unsupported_cited_lines,
    format_employment_evidence,
    is_comparison_request,
    normalize_undocumented_language,
)
from docwise.models import RetrievalDiagnostics, RetrievedChunk
from docwise.rag import RAGService
from docwise.retrieval import RetrievalOptions, Retriever


def chunk(text, source, index, score=0.8):
    return RetrievedChunk(
        document=Document(
            page_content=text,
            metadata={
                "source": source,
                "page": 1,
                "chunk_index": index,
                "chunk_id": f"{source}:{index}",
            },
        ),
        score=score,
    )


RESUME_SOURCES = [
    chunk(
        """WORK EXPERIENCE
Automation Solutions Developer (functional role) - Northstar Automation Ltd.
Pune, India
Dec 2025 - Sep 2026
Built Power Apps, SQL, Power BI and Power Automate solutions.""",
        "candidate_resume.pdf",
        0,
    ),
    chunk(
        """Data Administrator & Analyst - Harbor Property Services
Bristol, United Kingdom
Oct 2023 - Feb 2025
Cleaned and migrated records with Python, regex, SQLite, and SQL.
EDUCATION
Westbridge University Sep 2021 - Sep 2022""",
        "candidate_resume.pdf",
        1,
    ),
]

JD_SOURCE = chunk(
    """Job description: Data Analyst
Experience required: 3 to 5 years.
Mandatory skills: Python and SQL. Tableau and Power BI visualization experience.""",
    "data_analyst_jd.txt",
    0,
)


def test_employment_dates_are_parsed_sorted_and_totaled_without_llm_math():
    summary = extract_employment_history(
        [*RESUME_SOURCES, JD_SOURCE], as_of=date(2026, 10, 8)
    )

    assert len(summary.periods) == 2
    assert summary.professional_months == 25
    assert summary.professional_months >= 24
    assert summary.professional_is_approximate is True
    assert summary.professional_days is None
    assert summary.most_recent.company == "Northstar Automation Ltd."
    assert summary.most_recent.location == "Pune, India"
    assert summary.periods[1].company == "Harbor Property Services"


def test_structured_employment_evidence_has_source_backed_dates():
    sources = [*RESUME_SOURCES, JD_SOURCE]
    summary = extract_employment_history(sources, as_of=date(2026, 10, 8))
    evidence = format_employment_evidence(summary, sources)

    assert "Approximate non-overlapping duration for professional roles" in evidence
    assert "2 years, 1 month (25 months)" in evidence
    assert "exact day total unavailable" in evidence
    assert "Most recent listed role by end date" in evidence
    assert "Northstar Automation Ltd." in evidence
    assert "[1]" in evidence
    assert "[2]" in evidence


class ChunkStore:
    def __init__(self, chunks):
        self._chunks = chunks

    def all_chunks(self):
        return self._chunks


def raw_chunk(source, index, text):
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


def test_comparison_retrieval_covers_every_chunk_in_short_resume_and_jd():
    raw = [
        *[
            raw_chunk("candidate_resume.pdf", index, f"Resume evidence section {index}")
            for index in range(5)
        ],
        raw_chunk("data_analyst_jd.txt", 0, "Python SQL required 3 to 5 years"),
        raw_chunk("data_analyst_jd.txt", 1, "Power BI Tableau responsibilities"),
    ]
    retriever = Retriever(ChunkStore(raw))

    results, diagnostics = retriever.retrieve_for_comparison(
        "Compare candidate resume with the Data Analyst job description",
        RetrievalOptions(strategy="keyword", top_k=2),
    )

    assert len(results) == 7
    assert diagnostics.per_document == {
        "candidate_resume.pdf": 5,
        "data_analyst_jd.txt": 2,
    }
    assert diagnostics.comparison_mode is True
    assert diagnostics.truncated is False
    assert all(item.retrieval_reason == "complete short-document coverage" for item in results)


def test_context_budget_reports_omitted_chunks_instead_of_silent_truncation():
    raw = [
        raw_chunk("resume.pdf", 0, "resume " * 30),
        raw_chunk("resume.pdf", 1, "employment " * 30),
        raw_chunk("jd.txt", 0, "requirements " * 30),
    ]
    retriever = Retriever(ChunkStore(raw))
    results, diagnostics = retriever.retrieve_for_comparison(
        "Compare resume and job requirements",
        RetrievalOptions(
            strategy="keyword",
            top_k=1,
            max_context_characters=250,
        ),
    )
    assert len(results) < len(raw)
    assert diagnostics.truncated is True
    assert diagnostics.omitted_chunks


def test_single_document_requirement_question_does_not_force_comparison():
    assert is_comparison_request("List the requirements in the job description") is False
    assert is_comparison_request("How many years of experience does the JD require?") is False
    assert is_comparison_request("Compare my resume with the job requirements") is True


class ComparisonRetriever:
    def __init__(self, sources):
        self.sources = sources
        self.queries = []

    def retrieve_for_comparison(self, query, options):
        self.queries.append(query)
        diagnostics = RetrievalDiagnostics(
            query=query,
            strategy=options.strategy,
            comparison_mode=True,
            total_candidates=len(self.sources),
            selected_count=len(self.sources),
            per_document={"candidate_resume.pdf": 2, "data_analyst_jd.txt": 1},
            context_characters=sum(len(item.document.page_content) for item in self.sources),
        )
        return self.sources, diagnostics


class CapturingLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.response


def test_comparison_workflow_uses_structured_dates_and_explicit_source_citations(settings):
    sources = [*RESUME_SOURCES, JD_SOURCE]
    llm = CapturingLLM(
        "Employment history: Northstar Automation is the most recent role [1].\n"
        "Experience: the resume documents approximately 25 months of professional employment "
        "[1] [2].\n"
        "Requirement: the JD requests 3 to 5 years [3].\n"
        "Tableau is not documented in the resume [3]."
    )
    service = RAGService(settings, ComparisonRetriever(sources), llm)

    response = service.answer(
        "Compare my resume with the Data Analyst job description and identify gaps.",
        options=RetrievalOptions(strategy="hybrid", top_k=2),
    )

    system_prompt = llm.calls[0][0].content
    assert "2 years, 1 month (25 months)" in system_prompt
    assert "exact day total unavailable" in system_prompt
    assert "Most recent listed role by end date" in system_prompt
    assert "Never say the candidate lacks a skill" in system_prompt
    assert "candidate_resume.pdf, page 1, chunk 0" in response.answer
    assert "data_analyst_jd.txt, page 1, chunk 0" in response.answer
    assert response.diagnostics.comparison_mode is True


def test_citation_expansion_rejects_unknown_labels_and_names_exact_chunks():
    answer = expand_citations("Supported [1]. Unknown [9].", RESUME_SOURCES)
    assert "[1: candidate_resume.pdf, page 1, chunk 0]" in answer
    assert "[9]" not in answer


def test_absence_language_is_changed_to_not_documented():
    answer = normalize_undocumented_language(
        "The candidate lacks Tableau. Missing skills include seaborn."
    )
    assert "resume does not document Tableau" in answer
    assert "Skills not documented in the resume" in answer


def test_claim_with_unrelated_citation_is_suppressed():
    filtered, removed = filter_unsupported_cited_lines(
        "Supported: Northstar Automation is the employer [1].\n"
        "Unsupported: Kubernetes certification [1].",
        RESUME_SOURCES,
    )
    assert "Northstar Automation" in filtered
    assert "Kubernetes" not in filtered
    assert removed == 1


def test_exact_partial_month_dates_use_actual_elapsed_days():
    sources = [
        chunk(
            """WORK EXPERIENCE
Data Analyst - Example Ltd
Pune, India
15 Jan 2024 - 14 Feb 2024""",
            "exact_resume.pdf",
            0,
        )
    ]

    summary = extract_employment_history(sources)
    period = summary.periods[0]

    assert period.start_date == date(2024, 1, 15)
    assert period.end_date == date(2024, 2, 14)
    assert period.duration_days == 30
    assert period.duration_months is None
    assert period.is_approximate is False
    assert summary.professional_days == 30
    assert "30 days exact elapsed time" in format_employment_evidence(summary, sources)


def test_overlapping_exact_employment_is_counted_once():
    sources = [
        chunk(
            """WORK EXPERIENCE
Analyst - First Employer
Pune, India
2024-01-01 - 2024-03-01""",
            "overlap_resume.pdf",
            0,
        ),
        chunk(
            """Senior Analyst - Second Employer
Pune, India
2024-02-01 - 2024-04-01""",
            "overlap_resume.pdf",
            1,
        ),
    ]

    summary = extract_employment_history(sources)

    assert summary.professional_days == 91
    assert summary.professional_is_approximate is False
    assert summary.most_recent.company == "Second Employer"


def test_month_only_overlap_is_approximate_and_does_not_invent_days():
    sources = [
        chunk(
            """WORK EXPERIENCE
Analyst - First Employer
Pune, India
Oct 2023 - Feb 2025""",
            "month_resume.pdf",
            0,
        ),
        chunk(
            """Senior Analyst - Second Employer
Pune, India
Jan 2025 - Sep 2026""",
            "month_resume.pdf",
            1,
        ),
    ]

    summary = extract_employment_history(sources)

    assert summary.professional_months == 35
    assert summary.professional_days is None
    assert summary.professional_is_approximate is True
    assert all(period.start_date is None and period.end_date is None for period in summary.periods)
    assert all(period.is_approximate for period in summary.periods)


def test_paid_training_is_kept_separate_from_professional_duration():
    training = chunk(
        """WORK EXPERIENCE
Paid Training Program - Accenture
Bengaluru, India
Jan 2020 - Mar 2020""",
        "candidate_resume.pdf",
        2,
    )

    summary = extract_employment_history([*RESUME_SOURCES, training])

    assert summary.professional_months == 25
    assert summary.all_listed_months == 27
    assert any(period.is_training for period in summary.periods)
