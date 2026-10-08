"""Opt-in real-browser checks for the Streamlit portfolio experience.

Run with ``RUN_BROWSER_TESTS=1 pytest -m browser``. The test uses an isolated storage
directory and the synthetic fixture in ``tests/fixtures``; it never indexes local user data.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest
from dotenv import load_dotenv

pytestmark = pytest.mark.browser


def _browser_executable() -> str | None:
    configured = os.getenv("PLAYWRIGHT_BROWSER_EXECUTABLE")
    candidates = [
        configured,
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    return next((path for path in candidates if path and Path(path).is_file()), None)


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@contextmanager
def _streamlit_server(project_root: Path, storage: Path, *, password: str | None = None):
    port = _free_port()
    environment = os.environ.copy()
    environment.update(
        {
            "DOCWISE_STORAGE_DIR": str(storage),
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        }
    )
    if password:
        environment["DOCWISE_ACCESS_PASSWORD"] = password
    else:
        environment.pop("DOCWISE_ACCESS_PASSWORD", None)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless=true",
            "--server.address=127.0.0.1",
            f"--server.port={port}",
        ],
        cwd=project_root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + 90
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Streamlit exited before becoming healthy.")
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("Streamlit did not become healthy within 90 seconds.")
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _require_browser_environment() -> str:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    if os.getenv("RUN_BROWSER_TESTS") != "1":
        pytest.skip("Set RUN_BROWSER_TESTS=1 to run real browser tests.")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        pytest.skip("A Gemini credential is required for the live chat browser check.")
    executable = _browser_executable()
    if not executable:
        pytest.skip("Chrome or Edge was not found; set PLAYWRIGHT_BROWSER_EXECUTABLE.")
    return executable


def test_streamlit_document_chat_lifecycle_and_screenshots(tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    executable = _require_browser_environment()
    project_root = Path(__file__).resolve().parents[2]
    fixture = project_root / "tests" / "fixtures" / "demo_policy.md"
    screenshot_dir = Path(
        os.getenv("DOCWISE_SCREENSHOT_DIR", project_root / "test-results" / "screenshots")
    )
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    with _streamlit_server(project_root, tmp_path / "storage") as base_url:
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=executable, headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(base_url, wait_until="domcontentloaded", timeout=120_000)
            page.get_by_text("Chat with your documents").wait_for(timeout=120_000)

            # Friendly empty-state/error handling before any document exists.
            chat = page.get_by_placeholder("Ask a question about your indexed documents…")
            chat.fill("What is the Severity One response target?")
            chat.press("Enter")
            page.get_by_text("Upload and index at least one document").wait_for(timeout=30_000)
            page.get_by_role("button", name="Clear conversation").click()

            page.locator('input[type="file"]').set_input_files(str(fixture))
            page.get_by_text("demo_policy.md", exact=True).wait_for(timeout=30_000)
            page.wait_for_timeout(1_500)
            page.get_by_role("button", name="Index uploaded documents").click()
            try:
                page.get_by_text("Indexed 'demo_policy.md'").wait_for(timeout=120_000)
            except sync_api.TimeoutError as exc:
                raise AssertionError(
                    "Document indexing did not complete. Rendered UI:\n"
                    + page.locator("body").inner_text()[-6_000:]
                ) from exc
            page.get_by_text("Indexed documents (1)").wait_for(timeout=30_000)

            # Exercise model-management controls before generating a response.
            routing_mode = page.get_by_role("combobox", name="Routing mode")
            routing_mode.click()
            routing_mode.press("ArrowDown")
            routing_mode.press("ArrowDown")
            routing_mode.press("Enter")
            sync_api.expect(page.get_by_role("combobox", name="Routing mode")).to_have_value(
                "Quota Saver", timeout=30_000
            )
            model_restriction = page.get_by_role("combobox", name="Model restriction")
            model_restriction.click()
            model_restriction.press("End")
            model_restriction.press("Enter")
            sync_api.expect(
                page.get_by_role("combobox", name="Model restriction")
            ).to_have_value("gemini-3.1-flash-lite", timeout=30_000)

            chat = page.get_by_placeholder("Ask a question about your indexed documents…")
            chat.fill(
                "In one sentence, what is the acknowledgement target for a Severity One incident?"
            )
            chat.press("Enter")
            page.get_by_text("Generated using:").wait_for(timeout=180_000)
            assert page.get_by_text("15 minutes", exact=False).count() >= 1

            assistant_message = page.get_by_label("Chat message from assistant").last
            assistant_message.screenshot(
                path=str(screenshot_dir / "docwise-grounded-answer.png")
            )

            page.get_by_text("Sources (", exact=False).last.click()
            page.get_by_text("Retrieval diagnostics", exact=True).last.click()
            page.screenshot(
                path=str(screenshot_dir / "docwise-retrieval-diagnostics.png"),
                full_page=True,
            )

            # Conversation clearing and the full document lifecycle are browser-verified.
            page.get_by_role("button", name="Clear conversation").click()
            page.get_by_label("Chat message from user").last.wait_for(
                state="detached", timeout=30_000
            )
            page.get_by_text("demo_policy.md", exact=True).last.click()
            page.get_by_role("button", name="Delete").click()
            page.get_by_text("Indexed documents (0)").wait_for(timeout=30_000)
            browser.close()


def test_streamlit_optional_password_gate_and_screenshot(tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    executable = _require_browser_environment()
    project_root = Path(__file__).resolve().parents[2]
    screenshot_dir = Path(
        os.getenv("DOCWISE_SCREENSHOT_DIR", project_root / "test-results" / "screenshots")
    )
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    with _streamlit_server(
        project_root, tmp_path / "auth-storage", password="browser-test-password"
    ) as base_url:
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=executable, headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            page.goto(base_url, wait_until="domcontentloaded", timeout=60_000)
            page.get_by_text("Enter the deployment access password").wait_for(timeout=30_000)
            page.get_by_label("Access password").fill("incorrect")
            page.get_by_role("button", name="Sign in").click()
            page.get_by_text("Incorrect access password.").wait_for(timeout=30_000)
            page.screenshot(path=str(screenshot_dir / "docwise-access-gate.png"), full_page=True)
            page.get_by_label("Access password").fill("browser-test-password")
            page.get_by_role("button", name="Sign in").click()
            page.get_by_text("Chat with your documents").wait_for(timeout=120_000)
            browser.close()
