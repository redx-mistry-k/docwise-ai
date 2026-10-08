"""Small security controls for optional public-facing Streamlit deployments."""

from __future__ import annotations

import hmac
import os
import time
from dataclasses import dataclass


def access_password() -> str | None:
    """Read the optional access password without placing it in application settings or logs."""
    value = os.getenv("DOCWISE_ACCESS_PASSWORD", "")
    return value if value else None


def password_matches(candidate: str, expected: str) -> bool:
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def query_limit_per_minute() -> int:
    try:
        return min(120, max(1, int(os.getenv("DOCWISE_QUERY_RPM", "10"))))
    except ValueError:
        return 10


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    timestamps: tuple[float, ...]
    retry_after_seconds: int = 0


def check_session_rate_limit(
    prior_timestamps: list[float] | tuple[float, ...],
    *,
    limit: int,
    now: float | None = None,
) -> RateLimitDecision:
    """Apply a bounded per-session sliding-window limit without storing user identifiers."""
    current = time.monotonic() if now is None else now
    recent = tuple(value for value in prior_timestamps if current - value < 60)
    if len(recent) >= limit:
        retry = max(1, int(60 - (current - recent[0])))
        return RateLimitDecision(False, recent, retry)
    return RateLimitDecision(True, (*recent, current))
