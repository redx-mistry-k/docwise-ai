"""Free-tier-only Gemini routing, fallback, and persistent quota estimates."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from langchain_google_genai import ChatGoogleGenerativeAI

from .credentials import resolve_gemini_credential

from .exceptions import ConfigurationError, DocWiseError

PACIFIC = ZoneInfo("America/Los_Angeles")
APPROVED_FREE_MODELS = frozenset(
    {
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    }
)


class RoutingMode(StrEnum):
    BALANCED = "balanced"
    QUALITY_FIRST = "quality_first"
    QUOTA_SAVER = "quota_saver"


class TaskKind(StrEnum):
    QUERY_REWRITE = "query_rewrite"
    SIMPLE = "simple"
    REASONING = "reasoning"
    COMPARISON = "comparison"
    EVALUATION = "evaluation"


@dataclass(frozen=True, slots=True)
class ModelLimit:
    model_id: str
    tier: str
    rpm: int
    tpm: int
    rpd: int
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class RoutingPreferences:
    mode: RoutingMode = RoutingMode.BALANCED
    manual_selection: str | None = None
    priority_override: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class RoutedInvocation:
    message: Any
    model_id: str
    fallback_reason: str | None
    attempted_models: tuple[str, ...]
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ErrorDecision:
    kind: str
    retry_after_seconds: float
    detail: str


@dataclass(slots=True)
class ModelStatus:
    model_id: str
    attempted_today: int
    successful_today: int
    estimated_remaining: int
    cooldown_until: datetime | None
    daily_exhausted: bool
    eligible: bool
    reason: str | None = None
    next_available: datetime | None = None


@dataclass(frozen=True, slots=True)
class ModelRoutingConfig:
    models: dict[str, ModelLimit]
    priority: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> "ModelRoutingConfig":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Could not load Gemini routing configuration: {path}") from exc
        models: dict[str, ModelLimit] = {}
        for raw in payload.get("models", []):
            model_id = str(raw.get("id", ""))
            if model_id not in APPROVED_FREE_MODELS:
                raise ConfigurationError(
                    f"Model '{model_id}' is not in DocWise's approved free-tier allowlist."
                )
            tier = str(raw.get("tier", ""))
            if tier not in {"flash", "lite"}:
                raise ConfigurationError(f"Invalid tier for '{model_id}': {tier}")
            limit = ModelLimit(
                model_id=model_id,
                tier=tier,
                rpm=max(1, int(raw["rpm"])),
                tpm=max(1, int(raw["tpm"])),
                rpd=max(1, int(raw["rpd"])),
                enabled=bool(raw.get("enabled", True)),
            )
            models[model_id] = limit
        priority = tuple(str(value) for value in payload.get("priority", []))
        if not models or not priority:
            raise ConfigurationError("Gemini routing configuration must contain models and priority.")
        if set(priority) != set(models):
            raise ConfigurationError("Gemini priority must contain every configured model exactly once.")
        return cls(models=models, priority=priority)

    def validate_priority(self, priority: Iterable[str]) -> tuple[str, ...]:
        result = tuple(priority)
        if len(result) != len(set(result)) or set(result) != set(self.priority):
            raise ValueError("Priority must contain every approved configured model exactly once.")
        return result

    def candidates(self, task: TaskKind, preferences: RoutingPreferences) -> tuple[str, ...]:
        priority = self.validate_priority(preferences.priority_override or self.priority)
        enabled = tuple(model for model in priority if self.models[model].enabled)
        selection = preferences.manual_selection
        if selection == "lite_only":
            return tuple(model for model in enabled if self.models[model].tier == "lite")
        if selection:
            if selection not in APPROVED_FREE_MODELS or selection not in self.models:
                raise ConfigurationError(
                    f"Manual model '{selection}' is not an approved configured free-tier model."
                )
            start = enabled.index(selection) if selection in enabled else len(enabled)
            candidates = enabled[start:]
            if self.models[selection].tier == "lite":
                candidates = tuple(
                    model for model in candidates if self.models[model].tier == "lite"
                )
            return candidates
        if preferences.mode == RoutingMode.QUALITY_FIRST:
            return enabled
        if task in {TaskKind.QUERY_REWRITE, TaskKind.SIMPLE}:
            lite = tuple(model for model in enabled if self.models[model].tier == "lite")
            if preferences.mode == RoutingMode.QUOTA_SAVER:
                return lite
            strong = tuple(model for model in enabled if self.models[model].tier == "flash")
            return lite + strong
        strong = tuple(model for model in enabled if self.models[model].tier == "flash")
        lite = tuple(model for model in enabled if self.models[model].tier == "lite")
        return strong + lite


def routing_preferences_from_session_state(
    state: Mapping[str, Any],
    config: ModelRoutingConfig,
    default_mode: str = RoutingMode.BALANCED.value,
) -> RoutingPreferences:
    """Reconstruct immutable routing preferences from Streamlit-style session state."""
    raw_mode = str(state.get("gemini_routing_mode", default_mode))
    try:
        mode = RoutingMode(raw_mode)
    except ValueError:
        mode = RoutingMode.BALANCED
    raw_selection = state.get("gemini_manual_selection", "automatic")
    manual = None if raw_selection in {None, "automatic"} else str(raw_selection)
    raw_priority = state.get("gemini_priority", config.priority)
    if isinstance(raw_priority, str):
        values = tuple(
            value.strip()
            for value in raw_priority.replace(",", "\n").splitlines()
            if value.strip()
        )
    else:
        values = tuple(str(value) for value in raw_priority)
    priority = config.validate_priority(values)
    return RoutingPreferences(mode=mode, manual_selection=manual, priority_override=priority)


class UsageTracker:
    """Concurrency-safe local quota estimates stored in SQLite WAL mode."""

    def __init__(self, path: Path, clock: Callable[[], datetime] | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or (lambda: datetime.now(UTC))
        self._schema_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._schema_lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS daily_usage (
                    model_id TEXT NOT NULL,
                    quota_date TEXT NOT NULL,
                    attempted_requests INTEGER NOT NULL DEFAULT 0,
                    successful_requests INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    rate_limit_errors INTEGER NOT NULL DEFAULT 0,
                    daily_quota_errors INTEGER NOT NULL DEFAULT 0,
                    daily_exhausted INTEGER NOT NULL DEFAULT 0,
                    last_error_detail TEXT,
                    cooldown_until_utc TEXT,
                    PRIMARY KEY (model_id, quota_date)
                );
                CREATE TABLE IF NOT EXISTS request_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL UNIQUE,
                    answer_id TEXT,
                    purpose TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    quota_date TEXT NOT NULL,
                    attempted_at_utc TEXT NOT NULL,
                    completed_at_utc TEXT,
                    success INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    error_type TEXT,
                    error_detail TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_events_model_time
                    ON request_events(model_id, attempted_at_utc);
                CREATE TABLE IF NOT EXISTS router_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at_utc TEXT NOT NULL
                );
                """
            )

    def now(self) -> datetime:
        value = self.clock()
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    def quota_date(self, now: datetime | None = None) -> str:
        return (now or self.now()).astimezone(PACIFIC).date().isoformat()

    def next_reset(self, now: datetime | None = None) -> datetime:
        local = (now or self.now()).astimezone(PACIFIC)
        tomorrow = local.date() + timedelta(days=1)
        return datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=PACIFIC)

    def record_attempt(
        self,
        model_id: str,
        purpose: str,
        estimated_input_tokens: int,
        answer_id: str | None,
        limit: ModelLimit | None = None,
        ignore_cooldown: bool = False,
    ) -> str | None:
        now = self.now()
        quota_date = self.quota_date(now)
        request_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if limit is not None:
                daily = connection.execute(
                    "SELECT * FROM daily_usage WHERE model_id = ? AND quota_date = ?",
                    (model_id, quota_date),
                ).fetchone()
                if daily:
                    cooldown = _parse_datetime(daily["cooldown_until_utc"])
                    if (
                        int(daily["daily_exhausted"])
                        or int(daily["attempted_requests"]) >= limit.rpd
                        or (
                            not ignore_cooldown
                            and cooldown is not None
                            and cooldown > now
                        )
                    ):
                        connection.rollback()
                        return None
                threshold = (now.astimezone(UTC) - timedelta(seconds=60)).isoformat()
                rolling = connection.execute(
                    """SELECT COUNT(*) AS requests, COALESCE(SUM(input_tokens), 0) AS tokens
                       FROM request_events WHERE model_id = ? AND attempted_at_utc >= ?""",
                    (model_id, threshold),
                ).fetchone()
                if (
                    int(rolling["requests"]) >= limit.rpm
                    or int(rolling["tokens"]) + estimated_input_tokens > limit.tpm
                ):
                    connection.rollback()
                    return None
            connection.execute(
                """INSERT INTO daily_usage(model_id, quota_date, attempted_requests)
                   VALUES (?, ?, 1)
                   ON CONFLICT(model_id, quota_date) DO UPDATE SET
                     attempted_requests = attempted_requests + 1""",
                (model_id, quota_date),
            )
            connection.execute(
                """INSERT INTO request_events(
                     request_id, answer_id, purpose, model_id, quota_date,
                     attempted_at_utc, input_tokens
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    answer_id,
                    purpose,
                    model_id,
                    quota_date,
                    now.astimezone(UTC).isoformat(),
                    max(0, estimated_input_tokens),
                ),
            )
            connection.commit()
        return request_id

    def record_success(
        self,
        request_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        now = self.now().astimezone(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            event = connection.execute(
                "SELECT model_id, quota_date FROM request_events WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if not event:
                connection.rollback()
                return
            connection.execute(
                """UPDATE request_events SET success = 1, completed_at_utc = ?,
                     input_tokens = ?, output_tokens = ? WHERE request_id = ?""",
                (now, max(0, input_tokens), max(0, output_tokens), request_id),
            )
            connection.execute(
                """UPDATE daily_usage SET successful_requests = successful_requests + 1,
                     input_tokens = input_tokens + ?, output_tokens = output_tokens + ?,
                     cooldown_until_utc = NULL
                   WHERE model_id = ? AND quota_date = ?""",
                (max(0, input_tokens), max(0, output_tokens), event["model_id"], event["quota_date"]),
            )
            connection.commit()

    def record_error(
        self,
        request_id: str,
        kind: str,
        detail: str,
        cooldown_until: datetime | None = None,
    ) -> None:
        now = self.now().astimezone(UTC).isoformat()
        safe_detail = _safe_error_detail(detail)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            event = connection.execute(
                "SELECT model_id, quota_date FROM request_events WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if not event:
                connection.rollback()
                return
            connection.execute(
                """UPDATE request_events SET completed_at_utc = ?, error_type = ?,
                     error_detail = ? WHERE request_id = ?""",
                (now, kind, safe_detail, request_id),
            )
            rate_increment = 1 if kind in {"rpm", "tpm", "temporary_rate"} else 0
            daily_increment = 1 if kind == "daily_quota" else 0
            connection.execute(
                """UPDATE daily_usage SET
                     rate_limit_errors = rate_limit_errors + ?,
                     daily_quota_errors = daily_quota_errors + ?,
                     daily_exhausted = CASE WHEN ? = 1 THEN 1 ELSE daily_exhausted END,
                     last_error_detail = ?, cooldown_until_utc = COALESCE(?, cooldown_until_utc)
                   WHERE model_id = ? AND quota_date = ?""",
                (
                    rate_increment,
                    daily_increment,
                    daily_increment,
                    safe_detail,
                    cooldown_until.astimezone(UTC).isoformat() if cooldown_until else None,
                    event["model_id"],
                    event["quota_date"],
                ),
            )
            connection.commit()

    def set_cooldown(self, model_id: str, until: datetime, detail: str) -> None:
        quota_date = self.quota_date()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO daily_usage(model_id, quota_date, cooldown_until_utc, last_error_detail)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(model_id, quota_date) DO UPDATE SET
                     cooldown_until_utc = excluded.cooldown_until_utc,
                     last_error_detail = excluded.last_error_detail""",
                (model_id, quota_date, until.astimezone(UTC).isoformat(), _safe_error_detail(detail)),
            )
            connection.commit()

    def daily(self, model_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM daily_usage WHERE model_id = ? AND quota_date = ?",
                (model_id, self.quota_date()),
            ).fetchone()
        return dict(row) if row else {
            "model_id": model_id,
            "quota_date": self.quota_date(),
            "attempted_requests": 0,
            "successful_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "rate_limit_errors": 0,
            "daily_quota_errors": 0,
            "daily_exhausted": 0,
            "last_error_detail": None,
            "cooldown_until_utc": None,
        }

    def rolling_usage(self, model_id: str, seconds: int = 60) -> tuple[int, int]:
        threshold = (self.now().astimezone(UTC) - timedelta(seconds=seconds)).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS requests, COALESCE(SUM(input_tokens), 0) AS tokens
                   FROM request_events WHERE model_id = ? AND attempted_at_utc >= ?""",
                (model_id, threshold),
            ).fetchone()
        return int(row["requests"]), int(row["tokens"])

    def next_rate_availability(self, model_id: str, seconds: int = 60) -> datetime:
        threshold = (self.now().astimezone(UTC) - timedelta(seconds=seconds)).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                """SELECT MIN(attempted_at_utc) AS oldest FROM request_events
                   WHERE model_id = ? AND attempted_at_utc >= ?""",
                (model_id, threshold),
            ).fetchone()
        oldest = _parse_datetime(row["oldest"] if row else None)
        return (oldest + timedelta(seconds=seconds)) if oldest else self.now() + timedelta(seconds=seconds)

    def set_state(self, key: str, value: str | None) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO router_state(key, value, updated_at_utc) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                     updated_at_utc = excluded.updated_at_utc""",
                (key, value, self.now().astimezone(UTC).isoformat()),
            )

    def get_state(self, key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM router_state WHERE key = ?", (key,)
            ).fetchone()
        return str(row["value"]) if row and row["value"] is not None else None


class GeminiRoutingError(DocWiseError):
    """Raised for non-quota Gemini failures that should not be hidden by fallback."""


class NoEligibleModelError(DocWiseError):
    def __init__(self, next_available: datetime | None):
        self.next_available = next_available
        when = next_available.astimezone(PACIFIC).strftime("%Y-%m-%d %I:%M %p %Z") if next_available else "later"
        super().__init__(
            "All approved free-tier Gemini models are currently unavailable. "
            f"Estimated next availability: {when}. No paid model was used."
        )


class GeminiModelRouter:
    def __init__(
        self,
        config: ModelRoutingConfig,
        tracker: UsageTracker,
        client_factory: Callable[[str], Any],
        *,
        max_rate_retries: int = 2,
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 8.0,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self.tracker = tracker
        self.client_factory = client_factory
        self.max_rate_retries = max(0, max_rate_retries)
        self.base_backoff_seconds = max(0.0, base_backoff_seconds)
        self.max_backoff_seconds = max(self.base_backoff_seconds, max_backoff_seconds)
        self.sleeper = sleeper
        self._clients: dict[str, Any] = {}
        self._client_lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Any) -> "GeminiModelRouter":
        config_path = settings.model_config_path or settings.project_root / "config" / "gemini_models.json"
        usage_path = settings.usage_db_path or settings.storage_dir / "gemini_usage.sqlite3"
        config = ModelRoutingConfig.load(config_path)
        credential = resolve_gemini_credential()

        def factory(model_id: str) -> ChatGoogleGenerativeAI:
            return ChatGoogleGenerativeAI(
                model=model_id,
                google_api_key=credential.value,
                temperature=0,
                retries=0,
                request_timeout=settings.api_timeout_seconds,
            )

        router = cls(
            config,
            UsageTracker(usage_path),
            factory,
            max_rate_retries=settings.api_max_retries,
        )
        router.credential_source = credential.source
        router.multiple_credentials_configured = credential.both_configured
        router.credential_values_differ = credential.values_differ
        return router

    def _client(self, model_id: str) -> Any:
        with self._client_lock:
            if model_id not in self._clients:
                self._clients[model_id] = self.client_factory(model_id)
            return self._clients[model_id]

    def model_status(self, model_id: str, estimated_tokens: int = 0) -> ModelStatus:
        limit = self.config.models[model_id]
        daily = self.tracker.daily(model_id)
        now = self.tracker.now()
        cooldown = _parse_datetime(daily.get("cooldown_until_utc"))
        attempted = int(daily["attempted_requests"])
        remaining = max(0, limit.rpd - attempted)
        rolling_requests, rolling_tokens = self.tracker.rolling_usage(model_id)
        reason = None
        next_available = None
        if not limit.enabled:
            reason = "disabled"
        elif bool(daily["daily_exhausted"]) or remaining <= 0:
            reason = "estimated daily allowance exhausted"
            next_available = self.tracker.next_reset()
        elif cooldown and cooldown > now:
            reason = f"cooldown until {cooldown.astimezone(PACIFIC).strftime('%I:%M:%S %p %Z')}"
            next_available = cooldown
        elif rolling_requests >= limit.rpm:
            reason = "local RPM safety limit reached"
            next_available = self.tracker.next_rate_availability(model_id)
        elif rolling_tokens + estimated_tokens > limit.tpm:
            reason = "local TPM safety limit reached"
            next_available = self.tracker.next_rate_availability(model_id)
        return ModelStatus(
            model_id=model_id,
            attempted_today=attempted,
            successful_today=int(daily["successful_requests"]),
            estimated_remaining=remaining,
            cooldown_until=cooldown,
            daily_exhausted=bool(daily["daily_exhausted"]),
            eligible=reason is None,
            reason=reason,
            next_available=next_available,
        )

    def statuses(self) -> list[ModelStatus]:
        return [self.model_status(model) for model in self.config.priority]

    def select_model(
        self,
        task: TaskKind,
        preferences: RoutingPreferences,
        estimated_tokens: int = 0,
    ) -> str | None:
        for model_id in self.config.candidates(task, preferences):
            if self.model_status(model_id, estimated_tokens).eligible:
                return model_id
        return None

    def invoke_routed(
        self,
        messages: Any,
        *,
        purpose: str,
        task: TaskKind,
        preferences: RoutingPreferences | None = None,
        answer_id: str | None = None,
    ) -> RoutedInvocation:
        preferences = preferences or RoutingPreferences()
        candidates = self.config.candidates(task, preferences)
        if not candidates:
            raise NoEligibleModelError(self.tracker.next_reset())
        estimated_tokens = _estimate_input_tokens(messages)
        attempted_models: list[str] = []
        fallback_reasons: list[str] = []
        next_available: datetime | None = None

        for model_id in candidates:
            status = self.model_status(model_id, estimated_tokens)
            if not status.eligible:
                fallback_reasons.append(f"{model_id}: {status.reason}")
                if status.next_available:
                    next_available = _earliest(next_available, status.next_available)
                continue
            attempted_models.append(model_id)
            for retry_number in range(self.max_rate_retries + 1):
                request_id = self.tracker.record_attempt(
                    model_id,
                    purpose,
                    estimated_tokens,
                    answer_id,
                    self.config.models[model_id],
                    ignore_cooldown=retry_number > 0,
                )
                if request_id is None:
                    fallback_reasons.append(
                        f"{model_id}: became unavailable during concurrent quota reservation"
                    )
                    break
                try:
                    message = self._client(model_id).invoke(messages)
                    input_tokens, output_tokens = _usage_tokens(message, estimated_tokens)
                    self.tracker.record_success(request_id, input_tokens, output_tokens)
                    fallback_reason = "; ".join(fallback_reasons) or None
                    self.tracker.set_state("last_model_used", model_id)
                    self.tracker.set_state("last_fallback_reason", fallback_reason)
                    return RoutedInvocation(
                        message=message,
                        model_id=model_id,
                        fallback_reason=fallback_reason,
                        attempted_models=tuple(attempted_models),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                except Exception as exc:
                    decision = classify_api_error(exc, retry_number, self.base_backoff_seconds, self.max_backoff_seconds)
                    cooldown = None
                    if decision.kind in {"rpm", "tpm", "temporary_rate"}:
                        cooldown = self.tracker.now() + timedelta(seconds=decision.retry_after_seconds)
                    elif decision.kind == "model_unavailable":
                        cooldown = self.tracker.now() + timedelta(hours=1)
                    self.tracker.record_error(
                        request_id, decision.kind, decision.detail, cooldown
                    )
                    if decision.kind == "daily_quota":
                        fallback_reasons.append(f"{model_id}: daily request quota exhausted")
                        next_available = _earliest(next_available, self.tracker.next_reset())
                        break
                    if decision.kind in {"rpm", "tpm", "temporary_rate"}:
                        next_available = _earliest(next_available, cooldown)
                        if retry_number < self.max_rate_retries:
                            self.sleeper(decision.retry_after_seconds)
                            continue
                        fallback_reasons.append(
                            f"{model_id}: temporary {decision.kind.upper()} limit after bounded retries"
                        )
                        break
                    if decision.kind == "model_unavailable":
                        fallback_reasons.append(f"{model_id}: configured model unavailable")
                        next_available = _earliest(next_available, cooldown)
                        break
                    raise GeminiRoutingError(
                        f"Gemini request failed for approved model {model_id}: {decision.detail}"
                    ) from exc

        self.tracker.set_state("last_fallback_reason", "; ".join(fallback_reasons) or None)
        if next_available is None:
            next_available = self.tracker.next_reset()
        raise NoEligibleModelError(next_available)


def classify_api_error(
    error: Exception,
    retry_number: int = 0,
    base_backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 8.0,
) -> ErrorDecision:
    detail = _error_text(error)
    lowered = detail.lower()
    status = _error_status(error)
    retry_after = _retry_after(error)
    bounded = min(max_backoff_seconds, retry_after or base_backoff_seconds * (2**retry_number))

    daily_markers = (
        "requests_per_day",
        "requests per day",
        "perday",
        " rpd",
        "daily request quota",
        "daily quota",
        "generate_requests_per_model_per_day",
    )
    rpm_markers = ("requests_per_minute", "requests per minute", " rpm", "perminute")
    tpm_markers = (
        "tokens_per_minute",
        "tokens per minute",
        "inputtokenspermodelperminute",
        "tokenspermodelperminute",
        " tpm",
    )
    quota_signal_without_status = status is None and (
        "resource_exhausted" in lowered or "quota" in lowered
    )
    if status == 429 or quota_signal_without_status:
        if any(marker in lowered for marker in daily_markers):
            return ErrorDecision("daily_quota", 0, detail)
        if any(marker in lowered for marker in tpm_markers):
            return ErrorDecision("tpm", bounded, detail)
        if any(marker in lowered for marker in rpm_markers):
            return ErrorDecision("rpm", bounded, detail)
        return ErrorDecision("temporary_rate", bounded, detail)
    if status == 404 or "model not found" in lowered or "not_found" in lowered:
        return ErrorDecision("model_unavailable", 0, detail)
    if status in {408, 500, 502, 503, 504} or "timeout" in lowered or "deadline" in lowered:
        return ErrorDecision("unexpected", bounded, detail)
    return ErrorDecision("unexpected", 0, detail)


def _estimate_input_tokens(messages: Any) -> int:
    return max(1, len(str(messages)) // 4)


def _usage_tokens(message: Any, estimated_input: int) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", None) or {}
    if not usage and isinstance(response_metadata, dict):
        usage = response_metadata.get("usage_metadata", {}) or response_metadata.get("usage", {})
    input_tokens = usage.get("input_tokens", usage.get("prompt_token_count", estimated_input))
    output_tokens = usage.get("output_tokens", usage.get("candidates_token_count", 0))
    return max(0, int(input_tokens or estimated_input)), max(0, int(output_tokens or 0))


def _error_status(error: Exception) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _error_text(error: Exception) -> str:
    parts = [str(error)]
    for attribute in ("details", "response_json", "body"):
        value = getattr(error, attribute, None)
        if value:
            try:
                parts.append(json.dumps(value, default=str))
            except TypeError:
                parts.append(str(value))
    return _safe_error_detail(" | ".join(parts))


def _retry_after(error: Exception) -> float | None:
    value = getattr(error, "retry_after", None)
    if hasattr(value, "total_seconds"):
        value = value.total_seconds()
    try:
        if value is not None:
            return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    return None


def _safe_error_detail(detail: str) -> str:
    redacted = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "<redacted-api-key>", detail)
    return redacted[:1000]


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _earliest(first: datetime | None, second: datetime | None) -> datetime | None:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)
