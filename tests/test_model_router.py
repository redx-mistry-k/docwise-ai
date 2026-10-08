from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.documents import Document

from docwise.exceptions import ConfigurationError
from docwise.model_router import (
    GeminiModelRouter,
    GeminiRoutingError,
    ModelLimit,
    ModelRoutingConfig,
    NoEligibleModelError,
    RoutedInvocation,
    RoutingMode,
    RoutingPreferences,
    TaskKind,
    UsageTracker,
    classify_api_error,
    routing_preferences_from_session_state,
)
from docwise.models import RetrievedChunk
from docwise.rag import RAGService
from docwise.retrieval import RetrievalOptions

STRONG = "gemini-3.8-flash"
STRONG_2 = "gemini-3.7-flash"
LITE = "gemini-3.5-flash-lite"
LITE_2 = "gemini-3.1-flash-lite"


class MutableClock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class APIError(Exception):
    def __init__(self, status_code, detail, retry_after=None):
        super().__init__(detail)
        self.status_code = status_code
        self.details = {"quota": detail}
        self.retry_after = retry_after


class Message:
    def __init__(self, content="answer", input_tokens=20, output_tokens=5):
        self.content = content
        self.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


class SequenceClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def routing_config(rpd=20, rpm=5, tpm=250_000):
    models = {
        model: ModelLimit(
            model,
            "lite" if "lite" in model else "flash",
            rpm,
            tpm,
            500 if "lite" in model and rpd == 20 else rpd,
        )
        for model in (STRONG, STRONG_2, LITE, LITE_2)
    }
    return ModelRoutingConfig(models, (STRONG, STRONG_2, LITE, LITE_2))


def make_router(tmp_path, clients, *, config=None, clock=None, retries=0, sleeper=None):
    tracker = UsageTracker(tmp_path / "usage.sqlite3", clock=clock)
    router = GeminiModelRouter(
        config or routing_config(),
        tracker,
        lambda model: clients[model],
        max_rate_retries=retries,
        sleeper=sleeper or (lambda seconds: None),
    )
    return router, tracker


def test_smart_task_selection_and_lite_only_restriction(tmp_path):
    config = routing_config()
    tracker = UsageTracker(tmp_path / "usage.sqlite3")
    router = GeminiModelRouter(config, tracker, lambda model: SequenceClient([Message()]))

    assert router.select_model(TaskKind.SIMPLE, RoutingPreferences()) == LITE
    assert router.select_model(TaskKind.QUERY_REWRITE, RoutingPreferences()) == LITE
    assert router.select_model(TaskKind.COMPARISON, RoutingPreferences()) == STRONG
    assert router.select_model(
        TaskKind.SIMPLE, RoutingPreferences(mode=RoutingMode.QUALITY_FIRST)
    ) == STRONG
    assert router.select_model(
        TaskKind.SIMPLE,
        RoutingPreferences(mode=RoutingMode.QUOTA_SAVER, manual_selection="lite_only"),
    ) == LITE
    candidates = config.candidates(
        TaskKind.REASONING,
        RoutingPreferences(manual_selection=LITE),
    )
    assert candidates == (LITE, LITE_2)
    assert all("lite" in model for model in candidates)


def test_daily_quota_error_falls_back_in_priority_order(tmp_path):
    clients = {
        STRONG: SequenceClient([APIError(429, "GenerateRequestsPerModelPerDay exhausted")]),
        STRONG_2: SequenceClient([Message("fallback answer")]),
        LITE: SequenceClient([Message()]),
        LITE_2: SequenceClient([Message()]),
    }
    router, tracker = make_router(tmp_path, clients)

    result = router.invoke_routed(
        "question",
        purpose="answer_generation",
        task=TaskKind.COMPARISON,
        preferences=RoutingPreferences(mode=RoutingMode.QUALITY_FIRST),
        answer_id="answer-1",
    )

    assert result.model_id == STRONG_2
    assert result.attempted_models == (STRONG, STRONG_2)
    assert "daily request quota exhausted" in result.fallback_reason
    assert tracker.daily(STRONG)["daily_exhausted"] == 1


def test_rpm_limit_retries_same_model_before_switching(tmp_path):
    sleeps = []
    clients = {
        LITE: SequenceClient(
            [APIError(429, "GenerateRequestsPerMinute exceeded", retry_after=2), Message()]
        ),
        LITE_2: SequenceClient([Message()]),
        STRONG: SequenceClient([Message()]),
        STRONG_2: SequenceClient([Message()]),
    }
    router, tracker = make_router(
        tmp_path, clients, retries=1, sleeper=lambda seconds: sleeps.append(seconds)
    )

    result = router.invoke_routed(
        "simple question", purpose="answer_generation", task=TaskKind.SIMPLE
    )

    assert result.model_id == LITE
    assert result.attempted_models == (LITE,)
    assert clients[LITE].calls == 2
    assert clients[LITE_2].calls == 0
    assert sleeps == [2.0]
    assert tracker.daily(LITE)["rate_limit_errors"] == 1
    assert tracker.daily(LITE)["cooldown_until_utc"] is None


def test_tpm_error_sets_cooldown_and_uses_next_model(tmp_path):
    clients = {
        LITE: SequenceClient(
            [APIError(429, "GenerateContentInputTokensPerModelPerMinute exceeded")]
        ),
        LITE_2: SequenceClient([Message()]),
        STRONG: SequenceClient([Message()]),
        STRONG_2: SequenceClient([Message()]),
    }
    router, tracker = make_router(tmp_path, clients, retries=0)
    result = router.invoke_routed(
        "simple question", purpose="answer_generation", task=TaskKind.SIMPLE
    )

    assert result.model_id == LITE_2
    assert "TPM" in result.fallback_reason
    assert tracker.daily(LITE)["cooldown_until_utc"] is not None


def test_ambiguous_429_is_temporary_not_daily():
    decision = classify_api_error(APIError(429, "Resource exhausted, retry later"))
    assert decision.kind == "temporary_rate"
    assert decision.kind != "daily_quota"


def test_usage_persists_across_tracker_instances(tmp_path):
    path = tmp_path / "usage.sqlite3"
    first = UsageTracker(path)
    request_id = first.record_attempt(LITE, "query_rewrite", 12, None)
    first.record_success(request_id, 10, 4)

    second = UsageTracker(path)
    daily = second.daily(LITE)
    assert daily["attempted_requests"] == 1
    assert daily["successful_requests"] == 1
    assert daily["input_tokens"] == 10
    assert daily["output_tokens"] == 4


def test_pacific_daily_reset_uses_new_quota_period(tmp_path):
    clock = MutableClock(datetime(2026, 10, 9, 6, 59, tzinfo=UTC))  # 11:59 PM PDT
    tracker = UsageTracker(tmp_path / "usage.sqlite3", clock=clock)
    request_id = tracker.record_attempt(LITE, "answer_generation", 5, None)
    tracker.record_success(request_id, 5, 1)
    assert tracker.daily(LITE)["attempted_requests"] == 1
    assert tracker.quota_date() == "2026-10-08"

    clock.value = datetime(2026, 10, 9, 7, 1, tzinfo=UTC)  # 12:01 AM PDT
    assert tracker.quota_date() == "2026-10-09"
    assert tracker.daily(LITE)["attempted_requests"] == 0


def test_concurrent_usage_updates_are_not_lost(tmp_path):
    tracker = UsageTracker(tmp_path / "usage.sqlite3")

    def record(index):
        request_id = tracker.record_attempt(LITE, "answer_generation", 3, f"a-{index}")
        tracker.record_success(request_id, 3, 1)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(record, range(30)))

    daily = tracker.daily(LITE)
    assert daily["attempted_requests"] == 30
    assert daily["successful_requests"] == 30
    assert daily["input_tokens"] == 90
    assert daily["output_tokens"] == 30


def test_concurrent_router_requests_cannot_overshoot_daily_safety_limit(tmp_path):
    config = ModelRoutingConfig(
        {LITE: ModelLimit(LITE, "lite", rpm=100, tpm=1_000_000, rpd=5)},
        (LITE,),
    )

    class AlwaysClient:
        def invoke(self, messages):
            return Message()

    router, tracker = make_router(
        tmp_path,
        {LITE: AlwaysClient()},
        config=config,
    )

    def invoke(index):
        try:
            return router.invoke_routed(
                f"question {index}",
                purpose="answer_generation",
                task=TaskKind.SIMPLE,
            ).model_id
        except NoEligibleModelError:
            return None

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(invoke, range(20)))

    assert results.count(LITE) == 5
    assert tracker.daily(LITE)["attempted_requests"] == 5
    assert tracker.daily(LITE)["successful_requests"] == 5


def test_no_eligible_model_stops_without_paid_fallback(tmp_path):
    config = routing_config(rpd=1)
    clients = {model: SequenceClient([Message()]) for model in config.priority}
    router, tracker = make_router(tmp_path, clients, config=config)
    for model in config.priority:
        tracker.record_attempt(model, "test", 1, None)

    with pytest.raises(NoEligibleModelError, match="No paid model was used"):
        router.invoke_routed(
            "question",
            purpose="answer_generation",
            task=TaskKind.COMPARISON,
            preferences=RoutingPreferences(mode=RoutingMode.QUALITY_FIRST),
        )
    assert all(client.calls == 0 for client in clients.values())


def test_unexpected_api_error_is_not_hidden_by_fallback(tmp_path):
    clients = {
        STRONG: SequenceClient([APIError(500, "unexpected server failure")]),
        STRONG_2: SequenceClient([Message()]),
        LITE: SequenceClient([Message()]),
        LITE_2: SequenceClient([Message()]),
    }
    router, _ = make_router(tmp_path, clients)
    with pytest.raises(GeminiRoutingError, match=STRONG):
        router.invoke_routed(
            "question",
            purpose="answer_generation",
            task=TaskKind.COMPARISON,
            preferences=RoutingPreferences(mode=RoutingMode.QUALITY_FIRST),
        )
    assert clients[STRONG_2].calls == 0


def test_paid_or_unknown_model_is_rejected_from_config_and_manual_selection(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "priority": ["gemini-paid-pro"],
                "models": [
                    {
                        "id": "gemini-paid-pro",
                        "tier": "flash",
                        "rpm": 1,
                        "tpm": 1,
                        "rpd": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="allowlist"):
        ModelRoutingConfig.load(path)
    with pytest.raises(ConfigurationError, match="not an approved"):
        routing_config().candidates(
            TaskKind.SIMPLE,
            RoutingPreferences(manual_selection="gemini-paid-pro"),
        )


def test_streamlit_style_routing_state_survives_reconstruction():
    config = routing_config()
    state = {
        "gemini_routing_mode": "quota_saver",
        "gemini_manual_selection": "lite_only",
        "gemini_priority": "\n".join((LITE_2, LITE, STRONG_2, STRONG)),
    }
    first = routing_preferences_from_session_state(state, config)
    second = routing_preferences_from_session_state(dict(state), config)
    assert first == second
    assert first.mode == RoutingMode.QUOTA_SAVER
    assert first.manual_selection == "lite_only"
    assert first.priority_override[0] == LITE_2


def test_router_returns_usage_and_records_answer_identifier(tmp_path):
    clients = {
        LITE: SequenceClient([Message("ok", input_tokens=31, output_tokens=7)]),
        LITE_2: SequenceClient([Message()]),
        STRONG: SequenceClient([Message()]),
        STRONG_2: SequenceClient([Message()]),
    }
    router, tracker = make_router(tmp_path, clients)
    result = router.invoke_routed(
        "question",
        purpose="answer_generation",
        task=TaskKind.SIMPLE,
        answer_id="rag-answer-123",
    )
    assert result.model_id == LITE
    assert result.input_tokens == 31
    assert result.output_tokens == 7
    with sqlite3.connect(tracker.path) as connection:
        row = connection.execute(
            "SELECT model_id, answer_id FROM request_events WHERE answer_id = ?",
            ("rag-answer-123",),
        ).fetchone()
    assert row == (LITE, "rag-answer-123")


class StaticRetriever:
    def retrieve(self, query, options):
        return [
            RetrievedChunk(
                Document(
                    page_content="The policy provides twenty leave days.",
                    metadata={"source": "policy.pdf", "page": 1, "chunk_index": 0},
                ),
                score=0.9,
            )
        ]


class RoutedLLM:
    def invoke_routed(self, messages, **kwargs):
        return RoutedInvocation(
            Message("Twenty leave days [1]."),
            LITE,
            "strong model daily quota exhausted",
            (STRONG, LITE),
            20,
            5,
        )


def test_rag_response_reports_exact_generation_model(settings):
    service = RAGService(settings, StaticRetriever(), llm=RoutedLLM())
    response = service.answer("How much leave is provided?", options=RetrievalOptions())
    assert response.model_name == LITE
    assert response.fallback_reason == "Answer generation: strong model daily quota exhausted"


def test_model_availability_failure_follows_configured_priority(tmp_path):
    clients = {
        STRONG: SequenceClient([APIError(404, "configured model not found")]),
        STRONG_2: SequenceClient([Message("available fallback")]),
        LITE: SequenceClient([Message()]),
        LITE_2: SequenceClient([Message()]),
    }
    router, _ = make_router(tmp_path, clients)

    result = router.invoke_routed(
        "comparison",
        purpose="answer_generation",
        task=TaskKind.COMPARISON,
        preferences=RoutingPreferences(mode=RoutingMode.QUALITY_FIRST),
    )

    assert result.model_id == STRONG_2
    assert result.attempted_models == (STRONG, STRONG_2)
    assert "unavailable" in result.fallback_reason


def test_rewrite_answer_and_evaluation_purposes_are_all_usage_tracked(tmp_path):
    import sqlite3

    clients = {
        STRONG: SequenceClient([Message("comparison"), Message("grade")]),
        STRONG_2: SequenceClient([Message()]),
        LITE: SequenceClient([Message("rewrite"), Message("answer")]),
        LITE_2: SequenceClient([Message()]),
    }
    router, tracker = make_router(tmp_path, clients)

    router.invoke_routed("rewrite", purpose="query_rewrite", task=TaskKind.QUERY_REWRITE)
    router.invoke_routed("answer", purpose="answer_generation", task=TaskKind.SIMPLE)
    router.invoke_routed("grade", purpose="evaluation_grading", task=TaskKind.EVALUATION)

    with sqlite3.connect(tracker.path) as connection:
        purposes = {
            row[0] for row in connection.execute("SELECT purpose FROM request_events").fetchall()
        }
    assert purposes == {"query_rewrite", "answer_generation", "evaluation_grading"}
