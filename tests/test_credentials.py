from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from docwise.credentials import resolve_gemini_credential
from docwise.exceptions import ConfigurationError
from docwise.model_router import GeminiModelRouter


def test_prefers_canonical_google_key_when_both_are_present():
    selected = resolve_gemini_credential(
        {"GOOGLE_API_KEY": "google-secret", "GEMINI_API_KEY": "gemini-secret"}
    )

    assert selected.source == "GOOGLE_API_KEY"
    assert selected.value == "google-secret"
    assert selected.both_configured is True
    assert selected.values_differ is True
    assert "google-secret" not in repr(selected)
    assert "gemini-secret" not in repr(selected)


def test_accepts_gemini_api_key_as_compatibility_alias():
    selected = resolve_gemini_credential({"GEMINI_API_KEY": "alias-secret"})

    assert selected.source == "GEMINI_API_KEY"
    assert selected.value == "alias-secret"
    assert selected.both_configured is False


def test_missing_key_has_safe_actionable_error():
    with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY.*GEMINI_API_KEY") as error:
        resolve_gemini_credential({})

    assert "google-secret" not in str(error.value)
    assert "gemini-secret" not in str(error.value)


def test_router_passes_selected_key_explicitly(monkeypatch, settings):
    captured: dict[str, object] = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    project_config = Path(__file__).resolve().parents[1] / "config" / "gemini_models.json"
    routed_settings = replace(
        settings,
        model_config_path=project_config,
        usage_db_path=settings.storage_dir / "credential-test.sqlite3",
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "canonical-key")
    monkeypatch.setenv("GEMINI_API_KEY", "different-alias-key")
    monkeypatch.setattr("docwise.model_router.ChatGoogleGenerativeAI", fake_client)

    router = GeminiModelRouter.from_settings(routed_settings)
    router._client(router.config.priority[0])

    assert captured["google_api_key"] == "canonical-key"
    assert router.credential_source == "GOOGLE_API_KEY"
    assert router.multiple_credentials_configured is True
    assert router.credential_values_differ is True
