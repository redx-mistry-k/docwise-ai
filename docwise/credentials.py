"""Deterministic, secret-safe Gemini credential selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class GeminiCredential:
    """A selected credential whose value is intentionally hidden from repr/logs."""

    source: str
    value: str = ""
    both_configured: bool = False
    values_differ: bool = False

    def __repr__(self) -> str:
        return (
            "GeminiCredential("
            f"source={self.source!r}, value=<redacted>, "
            f"both_configured={self.both_configured!r}, "
            f"values_differ={self.values_differ!r})"
        )


def resolve_gemini_credential(
    environment: Mapping[str, str] | None = None,
) -> GeminiCredential:
    """Select a Gemini key without mutating the environment or exposing its value.

    ``GOOGLE_API_KEY`` is the canonical variable because it is the name documented by
    DocWise. ``GEMINI_API_KEY`` remains a supported compatibility alias. If both are
    present, the canonical variable wins explicitly, including when the values differ.
    """

    values = os.environ if environment is None else environment
    google_key = values.get("GOOGLE_API_KEY", "").strip()
    gemini_key = values.get("GEMINI_API_KEY", "").strip()
    both_configured = bool(google_key and gemini_key)

    if google_key:
        return GeminiCredential(
            source="GOOGLE_API_KEY",
            value=google_key,
            both_configured=both_configured,
            values_differ=both_configured and google_key != gemini_key,
        )
    if gemini_key:
        return GeminiCredential(source="GEMINI_API_KEY", value=gemini_key)
    raise ConfigurationError(
        "Missing Gemini credential. Set GOOGLE_API_KEY (preferred) or GEMINI_API_KEY "
        "in .env or the deployment secret settings."
    )
