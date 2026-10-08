"""Document-aware resume/job-description comparison helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .models import RetrievedChunk

DIRECT_COMPARISON_TERMS = re.compile(
    r"\b(compare|comparison|match|fit|align|alignment|gap|gaps|suitable|suitability|qualified|candidate)\b",
    re.IGNORECASE,
)
RESUME_TERMS = re.compile(
    r"\b(resume|résumé|cv|candidate (?:background|profile)|work history)\b",
    re.IGNORECASE,
)
JOB_TERMS = re.compile(
    r"\b(job description|jd|job|position|role requirements?|requirements?)\b",
    re.IGNORECASE,
)
DATE_RANGE_PATTERN = re.compile(
    r"(?P<start_month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+(?P<start_year>\d{4})\s*[-–—]\s*"
    r"(?:(?P<end_month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+(?P<end_year>\d{4})|(?P<present>Present|Current))",
    re.IGNORECASE,
)
_MONTH_NAME = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?"
)
EXACT_DATE_RANGE_PATTERNS = (
    re.compile(
        rf"(?P<start_day>\d{{1,2}})(?:st|nd|rd|th)?\s+"
        rf"(?P<start_month>{_MONTH_NAME})\s+(?P<start_year>\d{{4}})\s*[-–—]\s*"
        rf"(?:(?P<end_day>\d{{1,2}})(?:st|nd|rd|th)?\s+"
        rf"(?P<end_month>{_MONTH_NAME})\s+(?P<end_year>\d{{4}})|"
        rf"(?P<present>Present|Current))",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<start_month>{_MONTH_NAME})\s+"
        rf"(?P<start_day>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<start_year>\d{{4}})\s*[-–—]\s*"
        rf"(?:(?P<end_month>{_MONTH_NAME})\s+"
        rf"(?P<end_day>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<end_year>\d{{4}})|"
        rf"(?P<present>Present|Current))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<start_year>\d{4})-(?P<start_month>\d{2})-(?P<start_day>\d{2})\s*"
        r"(?:to|[-–—])\s*"
        r"(?:(?P<end_year>\d{4})-(?P<end_month>\d{2})-(?P<end_day>\d{2})|"
        r"(?P<present>Present|Current))",
        re.IGNORECASE,
    ),
)
MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
EVIDENCE_STOPWORDS = {
    "about", "against", "also", "and", "are", "candidate", "does", "documented",
    "from", "has", "have", "into", "its", "most", "not", "resume", "role", "that",
    "the", "their", "this", "used", "uses", "using", "was", "were", "with",
}


@dataclass(frozen=True, slots=True)
class EmploymentPeriod:
    role: str
    company: str
    location: str
    start_label: str
    end_label: str
    start_month_index: int
    end_month_index: int
    duration_months: int | None
    source: str
    page: int
    chunk_index: int
    chunk_id: str
    is_training: bool = False
    start_date: date | None = None
    end_date: date | None = None
    is_approximate: bool = True
    duration_days: int | None = None


@dataclass(slots=True)
class EmploymentSummary:
    periods: list[EmploymentPeriod] = field(default_factory=list)
    professional_months: int = 0
    all_listed_months: int = 0
    most_recent: EmploymentPeriod | None = None
    professional_days: int | None = None
    all_listed_days: int | None = None
    professional_is_approximate: bool = False
    all_listed_is_approximate: bool = False


def is_comparison_request(question: str, sources: list[RetrievedChunk] | None = None) -> bool:
    explicit_comparison = DIRECT_COMPARISON_TERMS.search(question) is not None
    paired_documents = RESUME_TERMS.search(question) is not None and JOB_TERMS.search(question) is not None
    candidate_job = (
        re.search(r"\bcandidate\b", question, re.IGNORECASE) is not None
        and JOB_TERMS.search(question) is not None
    )
    if not paired_documents and not (explicit_comparison and candidate_job):
        return False
    if sources is None:
        return True
    documents = {str(item.document.metadata.get("source", "")) for item in sources}
    return len(documents) >= 2


def looks_like_resume(source: str, text: str) -> bool:
    lowered_name = source.lower()
    lowered_text = text.lower()
    return (
        "resume" in lowered_name
        or re.search(r"(?:^|[_\-\s])cv(?:[._\-\s]|$)", lowered_name) is not None
        or "work experience" in lowered_text
        or "professional experience" in lowered_text
    )


def extract_employment_history(
    sources: list[RetrievedChunk],
    *,
    as_of: date | None = None,
) -> EmploymentSummary:
    """Parse resume work-history date ranges and calculate non-overlapping durations."""
    as_of = as_of or date.today()
    resume_names = {
        str(item.document.metadata.get("source", "Unknown"))
        for item in sources
        if looks_like_resume(
            str(item.document.metadata.get("source", "Unknown")),
            item.document.page_content,
        )
    }
    if not resume_names:
        return EmploymentSummary()

    candidates: dict[tuple[str, str, str, str], EmploymentPeriod] = {}
    for item in sources:
        metadata = item.document.metadata
        source = str(metadata.get("source", "Unknown"))
        if source not in resume_names:
            continue
        text = item.document.page_content
        education_position = re.search(r"\bEDUCATION\b", text, re.IGNORECASE)
        if education_position:
            text = text[: education_position.start()]
        matches: list[tuple[re.Match[str], bool]] = []
        exact_spans: list[tuple[int, int]] = []
        for pattern in EXACT_DATE_RANGE_PATTERNS:
            for exact_match in pattern.finditer(text):
                matches.append((exact_match, True))
                exact_spans.append(exact_match.span())
        for month_match in DATE_RANGE_PATTERN.finditer(text):
            if any(
                month_match.start() < end and month_match.end() > start
                for start, end in exact_spans
            ):
                continue
            matches.append((month_match, False))

        for match, has_exact_days in sorted(matches, key=lambda value: value[0].start()):
            preceding_lines = [
                line.strip()
                for line in text[: match.start()].splitlines()[-5:]
                if line.strip()
            ]
            role_line = next(
                (
                    line
                    for line in reversed(preceding_lines)
                    if " - " in line and len(line) <= 180
                ),
                None,
            )
            if not role_line:
                continue
            role, company = (part.strip() for part in role_line.rsplit(" - ", 1))
            role_position = preceding_lines.index(role_line)
            location_lines = preceding_lines[role_position + 1 :]
            location = location_lines[-1] if location_lines else "Not specified"

            start_month = _month_number(match.group("start_month"))
            start_year = int(match.group("start_year"))
            start_date = None
            end_date = None
            if has_exact_days:
                try:
                    start_date = date(start_year, start_month, int(match.group("start_day")))
                except ValueError:
                    continue
            if match.group("present"):
                end_month, end_year = as_of.month, as_of.year
                end_label = (
                    as_of.isoformat() + " (current)"
                    if has_exact_days
                    else as_of.strftime("%b %Y") + " (current)"
                )
                if has_exact_days:
                    end_date = as_of
            else:
                end_month = _month_number(match.group("end_month"))
                end_year = int(match.group("end_year"))
                if has_exact_days:
                    try:
                        end_date = date(end_year, end_month, int(match.group("end_day")))
                    except ValueError:
                        continue
                    end_label = end_date.isoformat()
                else:
                    end_label = f"{match.group('end_month')[:3].title()} {end_year}"
            start_index = start_year * 12 + start_month - 1
            end_index = end_year * 12 + end_month - 1
            if end_index < start_index:
                continue
            if has_exact_days and end_date < start_date:
                continue
            duration_days = (end_date - start_date).days if has_exact_days else None
            period = EmploymentPeriod(
                role=role,
                company=company,
                location=location,
                start_label=(
                    start_date.isoformat()
                    if has_exact_days
                    else f"{match.group('start_month')[:3].title()} {start_year}"
                ),
                end_label=end_label,
                start_month_index=start_index,
                end_month_index=end_index,
                duration_months=None if has_exact_days else end_index - start_index,
                source=source,
                page=int(metadata.get("page", 1)),
                chunk_index=int(metadata.get("chunk_index", 0)),
                chunk_id=str(metadata.get("chunk_id", "")),
                is_training="training" in f"{role} {company}".lower(),
                start_date=start_date,
                end_date=end_date,
                is_approximate=not has_exact_days,
                duration_days=duration_days,
            )
            key = (role.lower(), company.lower(), period.start_label, period.end_label)
            candidates[key] = period

    periods = sorted(
        candidates.values(),
        key=lambda period: (
            period.end_month_index,
            period.end_date.day if period.end_date else 0,
            period.start_month_index,
            period.start_date.day if period.start_date else 0,
        ),
        reverse=True,
    )
    professional = [period for period in periods if not period.is_training]
    professional_exact = bool(professional) and all(
        not period.is_approximate for period in professional
    )
    all_exact = bool(periods) and all(not period.is_approximate for period in periods)
    return EmploymentSummary(
        periods=periods,
        professional_months=_merged_month_count(professional),
        all_listed_months=_merged_month_count(periods),
        most_recent=periods[0] if periods else None,
        professional_days=_merged_day_count(professional) if professional_exact else None,
        all_listed_days=_merged_day_count(periods) if all_exact else None,
        professional_is_approximate=bool(professional) and not professional_exact,
        all_listed_is_approximate=bool(periods) and not all_exact,
    )


def format_employment_evidence(
    summary: EmploymentSummary,
    sources: list[RetrievedChunk],
) -> str:
    if not summary.periods:
        return "No employment date ranges could be parsed reliably from the retrieved resume evidence."
    citation_by_chunk = {
        str(item.document.metadata.get("chunk_id", "")): index
        for index, item in enumerate(sources, start=1)
    }
    lines = ["Deterministically parsed employment periods:"]
    for period in summary.periods:
        citation = citation_by_chunk.get(period.chunk_id)
        marker = f"[{citation}]" if citation else "[source unavailable]"
        category = "training" if period.is_training else "professional role"
        if period.is_approximate:
            duration = (
                f"approximately {format_duration(period.duration_months or 0)}; "
                f"day-level dates not provided"
            )
        else:
            duration = f"{period.duration_days} days exact elapsed time"
        lines.append(
            f"- {period.role} — {period.company}, {period.location}: "
            f"{period.start_label} to {period.end_label}; {duration}; {category} {marker}"
        )
    lines.append(_format_summary_duration("professional roles", summary, training=False))
    lines.append(
        _format_summary_duration(
            "professional roles plus explicitly labeled training", summary, training=True
        )
    )
    if summary.most_recent:
        recent = summary.most_recent
        citation = citation_by_chunk.get(recent.chunk_id)
        marker = f"[{citation}]" if citation else "[source unavailable]"
        lines.append(
            f"- Most recent listed role by end date: {recent.role} at {recent.company}, "
            f"{recent.location} ({recent.start_label} to {recent.end_label}) {marker}"
        )
    return "\n".join(lines)


def format_duration(months: int) -> str:
    years, remaining = divmod(months, 12)
    year_text = f"{years} year" + ("s" if years != 1 else "")
    month_text = f"{remaining} month" + ("s" if remaining != 1 else "")
    if years and remaining:
        return f"{year_text}, {month_text}"
    return year_text if years else month_text


def expand_citations(answer: str, sources: list[RetrievedChunk]) -> str:
    """Expand numeric labels into exact filename, page, and chunk references."""
    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index < 1 or index > len(sources):
            return ""
        metadata = sources[index - 1].document.metadata
        return (
            f"[{index}: {metadata.get('source', 'Unknown')}, "
            f"page {metadata.get('page', 1)}, chunk {metadata.get('chunk_index', 0)}]"
        )

    return re.sub(r"\[(\d+)\]", replace, answer).strip()


def normalize_undocumented_language(answer: str) -> str:
    """Avoid converting absence from a resume into a claim about candidate capability."""
    replacements = [
        (r"\bthe candidate lacks\b", "the resume does not document"),
        (r"\bcandidate lacks\b", "the resume does not document"),
        (r"\bthe candidate does not have\b", "the resume does not document"),
        (r"\bthe candidate does not possess\b", "the resume does not document"),
        (r"\bmissing skills\b", "skills not documented in the resume"),
        (r"\bmissing qualifications\b", "requirements not documented in the resume"),
    ]
    normalized = answer
    for pattern, replacement in replacements:
        normalized = re.sub(
            pattern,
            lambda match: replacement[0].upper() + replacement[1:]
            if match.group(0)[0].isupper()
            else replacement,
            normalized,
            flags=re.IGNORECASE,
        )
    return normalized


def filter_unsupported_cited_lines(
    answer: str,
    sources: list[RetrievedChunk],
) -> tuple[str, int]:
    """Remove generated claim lines with citations unrelated to their cited excerpts."""
    retained: list[str] = []
    removed = 0
    for line in answer.splitlines():
        labels = [int(value) for value in re.findall(r"\[(\d+)\]", line)]
        if not labels:
            retained.append(line)
            continue
        evidence = " ".join(
            sources[label - 1].document.page_content
            for label in labels
            if 1 <= label <= len(sources)
        )
        claim_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", re.sub(r"\[\d+\]", "", line.lower()))
            if len(token) > 2 and token not in EVIDENCE_STOPWORDS
        }
        evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence.lower()))
        derived_duration = (
            bool({"experience", "duration", "months", "years"} & claim_tokens)
            and _contains_date_range(evidence)
        )
        if claim_tokens & evidence_tokens or derived_duration:
            retained.append(line)
        else:
            removed += 1
    return "\n".join(retained).strip(), removed


def _merged_month_count(periods: list[EmploymentPeriod]) -> int:
    """Return an approximate month span without assuming full boundary months."""
    if not periods:
        return 0
    intervals = sorted(
        (period.start_month_index, period.end_month_index) for period in periods
    )
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _merged_day_count(periods: list[EmploymentPeriod]) -> int:
    """Return actual elapsed days across exact, non-overlapping date intervals."""
    intervals = sorted(
        (period.start_date, period.end_date)
        for period in periods
        if period.start_date is not None and period.end_date is not None
    )
    if not intervals:
        return 0
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += (current_end - current_start).days
            current_start, current_end = start, end
    return total + (current_end - current_start).days


def _format_summary_duration(
    label: str,
    summary: EmploymentSummary,
    *,
    training: bool,
) -> str:
    months = summary.all_listed_months if training else summary.professional_months
    days = summary.all_listed_days if training else summary.professional_days
    approximate = (
        summary.all_listed_is_approximate if training else summary.professional_is_approximate
    )
    if approximate:
        return (
            f"- Approximate non-overlapping duration for {label}: {format_duration(months)} "
            f"({months} months); exact day total unavailable because one or more ranges "
            f"omit day values."
        )
    return f"- Exact non-overlapping elapsed duration for {label}: {days or 0} days."


def _month_number(value: str) -> int:
    return int(value) if value.isdigit() else MONTHS[value[:3].lower()]


def _contains_date_range(text: str) -> bool:
    return DATE_RANGE_PATTERN.search(text) is not None or any(
        pattern.search(text) is not None for pattern in EXACT_DATE_RANGE_PATTERNS
    )
