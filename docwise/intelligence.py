"""Deterministic query planning, evidence assessment, and answer verification."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import RetrievedChunk
from .retrieval import tokenize

STOPWORDS = {
    "a", "about", "all", "an", "and", "are", "as", "at", "be", "by", "can",
    "compare", "describe", "did", "do", "does", "explain", "for", "from", "how",
    "i", "in", "include", "including", "is", "it", "its", "me", "more", "my",
    "of", "on", "or", "please", "that", "the", "their", "this", "to", "used",
    "using", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "would",
}
CLAIM_STOPWORDS = STOPWORDS | {
    "according", "answer", "candidate", "document", "documents", "evidence", "resume",
    "source", "sources",
}
FACETS = (
    ("technologies and tools", re.compile(r"\b(technolog(?:y|ies)|tools?|platforms?|stack)\b", re.I)),
    ("business problem and outcome", re.compile(r"\b(business problem|problem|outcome|impact|value)\b", re.I)),
    ("employment dates and duration", re.compile(r"\b(employment|experience|dates?|duration|years?|months?|recent role)\b", re.I)),
    ("required skills and qualifications", re.compile(r"\b(required|requirements?|qualifications?|mandatory|preferred)\b", re.I)),
    ("demonstrated skills", re.compile(r"\b(demonstrated|skills?|capabilities|proficiency)\b", re.I)),
    ("projects and responsibilities", re.compile(r"\b(projects?|responsibilities|duties|implemented|built)\b", re.I)),
    ("education and certifications", re.compile(r"\b(education|degree|certifications?|qualifications?)\b", re.I)),
    ("gaps or undocumented requirements", re.compile(r"\b(gaps?|missing|undocumented|not mentioned)\b", re.I)),
)
NEGATION_PATTERN = re.compile(r"\b(no|not|never|none|without|doesn't|didn't|cannot|can't)\b", re.I)
DATE_RANGE_PATTERN = re.compile(
    r"(?:\d{1,2}\s+)?[A-Za-z]{3,9}\s+\d{4}\s*[-–—]\s*"
    r"(?:(?:\d{1,2}\s+)?[A-Za-z]{3,9}\s+\d{4}|Present|Current)|"
    r"\d{4}-\d{2}-\d{2}\s*(?:to|[-–—])\s*(?:\d{4}-\d{2}-\d{2}|Present|Current)",
    re.I,
)


@dataclass(frozen=True, slots=True)
class QueryPlan:
    original_query: str
    subqueries: tuple[str, ...]
    complexity: str
    complexity_score: int
    adaptive_top_k: int
    decomposed: bool


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    sufficient: bool
    covered_subqueries: tuple[str, ...]
    uncovered_subqueries: tuple[str, ...]
    reason: str


@dataclass(slots=True)
class AnswerVerification:
    answer: str
    claims_checked: int = 0
    supported_claims: int = 0
    unsupported_claims: list[str] = field(default_factory=list)
    uncertain_claims: list[str] = field(default_factory=list)
    contradictory_claims: list[str] = field(default_factory=list)
    evidence_contradictions: list[str] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)

    @property
    def citation_correctness(self) -> float | None:
        if self.claims_checked == 0:
            return None
        return self.supported_claims / self.claims_checked


def plan_query(
    question: str,
    *,
    comparison: bool,
    base_top_k: int,
    max_top_k: int,
    max_subqueries: int,
    decompose: bool,
    adaptive: bool,
) -> QueryPlan:
    """Build a bounded local retrieval plan without spending an LLM call."""
    matched_facets = [label for label, pattern in FACETS if pattern.search(question)]
    separators = len(re.findall(r"[;?]|\b(?:and|also|plus)\b", question, re.I))
    complex_question = comparison or len(matched_facets) >= 2 or separators >= 2
    complexity_score = 3 if comparison and len(matched_facets) >= 2 else 2 if complex_question else 1
    complexity = {1: "simple", 2: "multi-part", 3: "document-comparison"}[complexity_score]
    top_k = base_top_k
    if adaptive:
        top_k = max(base_top_k, min(max_top_k, base_top_k + (complexity_score - 1) * 2))

    subqueries = [question]
    if decompose and complex_question:
        subject = _subject_terms(question)
        for facet in matched_facets:
            focused = f"{subject} {facet}".strip()
            if focused and focused.lower() not in {value.lower() for value in subqueries}:
                subqueries.append(focused)
            if len(subqueries) >= max_subqueries:
                break
        if len(subqueries) == 1:
            for clause in _split_clauses(question):
                if clause.lower() not in {value.lower() for value in subqueries}:
                    subqueries.append(clause)
                if len(subqueries) >= max_subqueries:
                    break

    return QueryPlan(
        original_query=question,
        subqueries=tuple(subqueries[:max_subqueries]),
        complexity=complexity,
        complexity_score=complexity_score,
        adaptive_top_k=top_k,
        decomposed=len(subqueries) > 1,
    )


def assess_evidence(
    plan: QueryPlan,
    sources: list[RetrievedChunk],
    *,
    minimum_score: float,
    subquery_matches: dict[str, int] | None = None,
    comparison: bool = False,
) -> EvidenceAssessment:
    if not sources:
        return EvidenceAssessment(False, (), plan.subqueries, "No chunks were retrieved.")
    if not comparison and max(item.score for item in sources) < minimum_score:
        return EvidenceAssessment(
            False,
            (),
            plan.subqueries,
            "The best retrieved chunk was below the configured relevance threshold.",
        )

    covered: list[str] = []
    uncovered: list[str] = []
    evidence_tokens = set(tokenize(" ".join(item.document.page_content for item in sources)))
    for query in plan.subqueries:
        match_count = (subquery_matches or {}).get(query, 0)
        query_tokens = _meaningful_tokens(query)
        lexical_coverage = len(query_tokens & evidence_tokens) / max(len(query_tokens), 1)
        if match_count > 0 or lexical_coverage >= 0.25:
            covered.append(query)
        else:
            uncovered.append(query)

    documents = {
        str(item.document.metadata.get("source", "Unknown")) for item in sources
    }
    document_coverage = not comparison or len(documents) >= 2
    required_coverage = 0.5 if len(plan.subqueries) > 1 else 1.0
    coverage = len(covered) / max(len(plan.subqueries), 1)
    sufficient = document_coverage and coverage >= required_coverage
    if not document_coverage:
        reason = "A multi-document comparison did not retrieve evidence from at least two files."
    elif not sufficient:
        reason = "One or more focused parts of the question lacked relevant retrieved evidence."
    else:
        reason = "Retrieved evidence met the local relevance and coverage checks."
    return EvidenceAssessment(
        sufficient,
        tuple(covered),
        tuple(uncovered),
        reason,
    )


def reformulate_query(plan: QueryPlan, assessment: EvidenceAssessment) -> str:
    """Create one deterministic retry query from uncovered facets and salient terms."""
    source = " ".join(assessment.uncovered_subqueries) or plan.original_query
    terms = list(dict.fromkeys(tokenize(source)))
    meaningful = [term for term in terms if term not in STOPWORDS]
    expansions: list[str] = []
    joined = " ".join(meaningful)
    if re.search(r"\b(employment|experience|role)\b", joined):
        expansions.extend(["work history", "start end dates"])
    if re.search(r"\b(technology|technologies|tools|stack)\b", joined):
        expansions.extend(["software platforms", "technical implementation"])
    if re.search(r"\b(requirement|required|qualification)\b", joined):
        expansions.extend(["mandatory preferred skills", "job description"])
    return " ".join([*meaningful[:20], *expansions]).strip() or plan.original_query


def verify_answer(
    answer: str,
    sources: list[RetrievedChunk],
    *,
    comparison: bool = False,
) -> AnswerVerification:
    """Validate claim citations against their actual chunks using conservative local checks."""
    retained: list[str] = []
    report = AnswerVerification(
        answer="",
        evidence_contradictions=detect_evidence_contradictions(sources),
    )
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line or _is_structural_line(line):
            retained.append(raw_line)
            continue
        labels = [
            int(value)
            for group in re.findall(r"\[([\d,\s]+)\]", line)
            for value in re.findall(r"\d+", group)
        ]
        invalid = [label for label in labels if label < 1 or label > len(sources)]
        report.invalid_citations.extend(invalid)
        valid = [label for label in labels if 1 <= label <= len(sources)]
        if not _looks_like_claim(line):
            retained.append(raw_line)
            continue

        report.claims_checked += 1
        evidence_chunks = [sources[label - 1].document.page_content for label in valid]
        states = [
            _claim_support_state(line, evidence, comparison=comparison)
            for evidence in evidence_chunks
        ]
        if "supported" in states:
            report.supported_claims += 1
            retained.append(raw_line)
        else:
            clean_claim = re.sub(
                r"\s+", " ", re.sub(r"\[[\d,\s]+\]", "", line)
            ).strip()
            if "contradictory" in states:
                report.contradictory_claims.append(clean_claim)
            elif "uncertain" in states:
                report.uncertain_claims.append(clean_claim)
                retained.append(raw_line)
            else:
                report.unsupported_claims.append(clean_claim)

    report.answer = "\n".join(retained).strip()
    report.invalid_citations = sorted(set(report.invalid_citations))
    return report


def _claim_support_state(claim: str, evidence: str, *, comparison: bool) -> str:
    uncited_claim = re.sub(r"\[[\d,\s]+\]", "", claim)
    claim_tokens = _meaningful_tokens(uncited_claim, CLAIM_STOPWORDS)
    evidence_tokens = _meaningful_tokens(evidence, CLAIM_STOPWORDS)
    overlap = claim_tokens & evidence_tokens
    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", uncited_claim))
    evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence))
    derived_duration = (
        comparison
        and bool(re.search(r"\b(days?|months?|years?|duration|experience)\b", claim, re.I))
        and DATE_RANGE_PATTERN.search(evidence) is not None
    )
    if claim_numbers - evidence_numbers and not derived_duration:
        return "unsupported"
    if "not documented" in claim.lower() and comparison:
        return "supported"
    if overlap and _is_contradictory(claim, evidence):
        return "contradictory"
    if overlap:
        return "supported"
    if evidence:
        return "uncertain"
    return "unsupported"


def _is_contradictory(claim: str, evidence: str) -> bool:
    claim_negated = NEGATION_PATTERN.search(claim) is not None
    evidence_negated = NEGATION_PATTERN.search(evidence) is not None
    shared = _meaningful_tokens(claim, CLAIM_STOPWORDS) & _meaningful_tokens(
        evidence, CLAIM_STOPWORDS
    )
    return len(shared) >= 2 and claim_negated != evidence_negated


def detect_evidence_contradictions(sources: list[RetrievedChunk]) -> list[str]:
    """Flag strong lexical contradictions between retrieved chunks for human review."""
    contradictions: list[str] = []
    for first_index, first in enumerate(sources):
        first_text = first.document.page_content
        first_tokens = _meaningful_tokens(first_text, CLAIM_STOPWORDS)
        for second in sources[first_index + 1 :]:
            second_text = second.document.page_content
            shared = first_tokens & _meaningful_tokens(second_text, CLAIM_STOPWORDS)
            if len(shared) < 3:
                continue
            polarity_conflict = bool(NEGATION_PATTERN.search(first_text)) != bool(
                NEGATION_PATTERN.search(second_text)
            )
            first_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", first_text))
            second_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", second_text))
            numeric_conflict = bool(first_numbers and second_numbers and first_numbers != second_numbers)
            if not polarity_conflict and not numeric_conflict:
                continue
            first_name = str(first.document.metadata.get("source", "Unknown"))
            second_name = str(second.document.metadata.get("source", "Unknown"))
            contradictions.append(
                f"Potential conflict between {first_name} and {second_name} on: "
                + ", ".join(sorted(shared)[:6])
            )
    return contradictions


def _subject_terms(question: str) -> str:
    terms = [term for term in tokenize(question) if term not in STOPWORDS]
    return " ".join(list(dict.fromkeys(terms))[:12])


def _split_clauses(question: str) -> list[str]:
    values = re.split(r"\s*(?:[;?]|\band\b|\balso\b|\bplus\b)\s*", question, flags=re.I)
    return [value.strip(" .,?") for value in values if len(tokenize(value)) >= 3]


def _meaningful_tokens(text: str, stopwords: set[str] = STOPWORDS) -> set[str]:
    return {token for token in tokenize(text) if len(token) > 2 and token not in stopwords}


def _is_structural_line(line: str) -> bool:
    plain = re.sub(r"^[#>*\-\d.\s]+", "", line).strip()
    return (
        not plain
        or (plain.endswith(":") and len(tokenize(plain)) <= 8)
        or (line.startswith("#") and len(tokenize(plain)) <= 10)
    )


def _looks_like_claim(line: str) -> bool:
    plain = re.sub(r"^[>*\-\d.\s]+", "", line).strip()
    return len(tokenize(plain)) >= 3 and plain != "I couldn't find that information in the indexed documents."
