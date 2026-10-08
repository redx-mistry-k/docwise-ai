from pathlib import Path

from docwise.document_loader import safe_filename
from docwise.security import check_session_rate_limit, password_matches
from docwise.vector_store import VectorStoreManager


def test_access_password_uses_constant_time_comparison_contract():
    assert password_matches("correct horse", "correct horse") is True
    assert password_matches("wrong", "correct horse") is False


def test_session_rate_limit_is_bounded_and_recovers_after_window():
    first = check_session_rate_limit([], limit=2, now=100.0)
    second = check_session_rate_limit(first.timestamps, limit=2, now=110.0)
    blocked = check_session_rate_limit(second.timestamps, limit=2, now=120.0)
    recovered = check_session_rate_limit(blocked.timestamps, limit=2, now=161.0)

    assert first.allowed is True
    assert second.allowed is True
    assert blocked.allowed is False
    assert blocked.retry_after_seconds == 40
    assert recovered.allowed is True


def test_traversal_filename_is_reduced_to_a_safe_basename():
    assert safe_filename(r"..\..\private\candidate?.pdf") == "candidate_.pdf"


def test_delete_refuses_to_remove_an_unmanaged_source_file(settings, embeddings, tmp_path):
    source = tmp_path / "outside-managed-storage.txt"
    source.write_text("Private source content that should remain after index deletion.", encoding="utf-8")
    manager = VectorStoreManager(settings, embeddings=embeddings)
    indexed = manager.index_file(source, copy_to_storage=False)

    manager.delete_document(indexed.record.document_id)

    assert source.exists()
    assert manager.list_documents() == []


def test_gitignore_excludes_secrets_private_storage_and_usage_database():
    root = Path(__file__).resolve().parents[1]
    patterns = (root / ".gitignore").read_text(encoding="utf-8")

    assert ".env" in patterns
    assert ".streamlit/secrets.toml" in patterns
    assert ".docwise/" in patterns
    assert "data/*" in patterns
    assert "*.sqlite" in patterns or ".docwise/" in patterns
