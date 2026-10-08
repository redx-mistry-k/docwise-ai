# DocWise AI Reliability and Security Audit

Audit date: 2026-10-09

## 1. Current architecture

Documents pass through extension/size validation, page-aware extraction, deterministic chunking, local sentence-transformer embeddings, and persistent Chroma storage. Queries pass through conversational reference resolution, local complexity planning, vector/BM25/hybrid retrieval, optional decomposition and adaptive depth, evidence sufficiency checks, and one bounded deterministic retry. Gemini generation uses the free-tier allowlisted router and persistent SQLite usage tracking. Generated claims and citations are then checked against retrieved chunks before Streamlit or the CLI displays the answer and diagnostics.

## 2. Automated and application tests

- Default Pytest suite: **75 passed, 2 opt-in browser tests skipped**.
- Python compilation: passed for application and tests.
- Dependency consistency: `pip check` reported no broken requirements.
- Streamlit server: started successfully; `/_stcore/health` returned HTTP 200 and the application shell returned HTTP 200.
- Playwright/Chrome suite: **2 passed** against isolated synthetic data. It verified the password
  gate, invalid-password handling, upload, indexing, model-mode/model selection, live chat,
  response rendering, sources, retrieval diagnostics, conversation clearing, and deletion.
  Screenshots are stored under `docs/screenshots/`.
- Dockerfile review passed, but an image build could not run because the local Docker Desktop Linux daemon was not running.

The suite covers local file-to-answer flow, TXT/Markdown/PDF loading behavior, chunking, deterministic embeddings, real temporary Chroma persistence, vector/BM25/hybrid retrieval, long-document evidence, comparisons, conversational follow-ups, ambiguous numbered references, unsupported questions, verifier behavior, contradictions, invalid citations, duplicate uploads, deletion, reindexing, model fallback, usage persistence, access checks, and session rate limiting.

## 3. Retrieval benchmark

The reproducible benchmark contains 41 manually labeled questions over 10 synthetic documents. Four questions intentionally have no relevant source and are reserved for refusal evaluation, leaving 37 retrieval-scored examples. Labels identify source files and exact expected evidence text.

Top-k 2 results:

| Strategy | Recall@K | Precision@K | MRR | Mean local latency |
|---|---:|---:|---:|---:|
| Baseline vector | 0.86 | 0.49 | 0.88 | 10.20 ms |
| BM25 | 0.85 | 0.49 | 0.91 | 0.24 ms |
| Hybrid | 0.86 | 0.50 | 0.92 | 10.38 ms |
| Hybrid + reranking | 0.86 | 0.50 | 0.92 | 10.23 ms |
| Intelligent adaptive retrieval | 0.88 | 0.50 | 0.92 | 11.51 ms |

Top-k 4 results:

| Strategy | Recall@K | Precision@K | MRR | Mean local latency |
|---|---:|---:|---:|---:|
| Baseline vector | 0.93 | 0.28 | 0.89 | 11.11 ms |
| BM25 | 0.92 | 0.28 | 0.91 | 0.24 ms |
| Hybrid | 0.93 | 0.28 | 0.93 | 10.80 ms |
| Hybrid + reranking | 0.93 | 0.28 | 0.93 | 10.94 ms |
| Intelligent adaptive retrieval | 0.93 | 0.28 | 0.93 | 12.25 ms |

At top-k 2, adaptive retrieval improved recall by 0.02 over hybrid reranking with unchanged precision and MRR. At top-k 4 it produced no quality gain and added latency. No natural benchmark question triggered the retry; deterministic tests separately verify successful retry recovery and final refusal.

## 4. Gemini routing verification

Automated tests verify Balanced, Quality First, Quota Saver, Lite-only/manual selection, daily exhaustion, RPM retry, TPM cooldown, unavailable-model fallback, configured priority, process-restart persistence, concurrent quota accounting, response attribution, and rejection of paid or unknown models. Query rewrite, answer generation, and evaluation-grading purposes are present in SQLite request events.

Two bounded live calls succeeded using `gemini-3.1-flash-lite`: one direct router smoke call and one full synthetic-document RAG call with one retrieved source, a valid citation, and one tracked Gemini call. No quota exhaustion was attempted live. Local limits remain estimates and do not prove Google-side quota or zero billing.

## 5. Security findings and fixes

- Added optional constant-time password access control through `DOCWISE_ACCESS_PASSWORD`.
- Added configurable per-session request limiting through `DOCWISE_QUERY_RPM`.
- Explicitly enabled Streamlit XSRF protection.
- Docker now runs as an unprivileged `docwise` user.
- Confirmed upload extension, size, empty-file, corrupt-PDF, encrypted/scanned-PDF, and filename sanitization behavior.
- Confirmed managed-file deletion cannot remove an external source path.
- Confirmed document prompt injection remains untrusted evidence in prompts and tests.
- Secret scan found no key-like literal outside excluded local-secret/storage paths; the README placeholder is intentional.
- `.env`, Streamlit secrets, `.docwise`, Chroma data, virtual environments, caches, and `data/*` are ignored.
- The environment defines both `GOOGLE_API_KEY` and `GEMINI_API_KEY` with different values.
  DocWise now resolves this deterministically: `GOOGLE_API_KEY` is canonical, the alias is used only
  when the canonical variable is absent, and the selected value is passed explicitly to the SDK.
  Neither stored credential was revealed, modified, or deleted.
- Git was initialized locally. Ignore checks protect `.env`, Streamlit secrets, runtime storage,
  uploaded files, virtual environments, test output, and local databases. The candidate release set
  was scanned for key patterns and personal identifiers before staging; user-specific resume labels
  in regression fixtures were replaced with fictional data.

The built-in password and session limiter are not sufficient for a high-risk public deployment. Use an identity-aware proxy, TLS, server-side user authorization, centralized rate limiting, and separate per-environment API projects.

## 6. Performance and reliability

- BM25 corpus tokenization and document frequency are cached per retriever, avoiding repeated full-corpus work during decomposition and retry.
- Embedding model initialization remains process-cached.
- Advanced decomposition, sufficiency checking, reformulation, and verification use no Gemini calls.
- Streamlit retains at most 40 UI messages; model conversation history remains separately bounded.
- Context is constrained by chunk count, characters, and estimated tokens.
- Comparison document routing now selects relevant documents instead of expanding every short indexed file.
- Chroma scores now use explicit cosine-distance conversion, eliminating weak-match relevance warnings.

## 7. Live versus mocked verification

Verified live:

- One direct free-tier Lite Gemini request.
- One synthetic-document end-to-end RAG generation with retrieval and citation.
- Streamlit process startup, health endpoint, and HTML shell.

Verified with deterministic/local tests or mocks:

- Quota exhaustion and fallback modes.
- RPM/TPM behavior and unavailable models.
- Most answer-generation regressions, conversational rewrites, contradiction handling, and refusal behavior.

Verified with a real local browser and synthetic data:

- Streamlit access gate and invalid-password message.
- File upload, indexing, grounded live response, source/diagnostic expansion, routing-control change,
  conversation clearing, and document deletion.

## 8. Remaining risks and limitations

- Deterministic verification cannot prove semantic entailment. Lexically uncertain cited paraphrases are preserved and flagged to avoid destructive false rejection.
- Contradiction detection is heuristic and may miss nuanced or temporally scoped disagreements.
- Estimated token counts use a character approximation.
- The benchmark is synthetic and portfolio-scale; it is not evidence of production performance at millions of chunks.
- Session-level rate limiting can be bypassed with new sessions.
- OCR, multi-tenant authorization, malware scanning, centralized audit logs, and a transactional multi-instance registry are not included.
- A successful free-tier request does not guarantee future quota availability or absence of charges.

## 9. Commands

```powershell
# Tests and compilation
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app.py chat.py evaluate.py docwise tests
.\.venv\Scripts\python.exe -m pip check

# Safe local UI
.\.venv\Scripts\streamlit.exe run app.py --server.address=127.0.0.1

# Reproducible benchmark
$env:HF_HUB_OFFLINE='1'       # optional after the embedding model is cached
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe evaluate.py `
  --dataset evaluation\docwise_benchmark_40.json `
  --corpus-dir evaluation\benchmark_corpus `
  --top-k 2 --chunk-size 350 --chunk-overlap 60 `
  --output .docwise\benchmark_report_top2.json

# Docker (start Docker Desktop first)
docker build -t docwise-ai .
docker run --rm -p 8501:8501 `
  -e GOOGLE_API_KEY=$env:GOOGLE_API_KEY `
  -e DOCWISE_ACCESS_PASSWORD=$env:DOCWISE_ACCESS_PASSWORD `
  -v "${PWD}\docwise-storage:/app/storage" `
  docwise-ai

# Pre-commit secret/tracking review after Git initialization
git status --short
git check-ignore -v .env .docwise data
git grep -n -I -E "AIza[0-9A-Za-z_-]{20,}"
```

## 10. GitHub and deployment checklist

- [x] Pinned dependencies and reproducible setup instructions
- [x] Architecture diagram and accurate feature documentation
- [x] Synthetic benchmark corpus, labels, commands, and measured results
- [x] `.env.example`, secure `.gitignore`, and `.dockerignore`
- [x] Non-root Docker runtime and health check
- [x] Optional access password and basic session rate limit
- [x] Known limitations and troubleshooting guidance
- [x] Initialize Git and verify ignored/tracked files
- [x] Capture local browser screenshots with the automated synthetic-data flow
- [ ] Start Docker Desktop and complete an image build/run smoke test
- [ ] Configure production identity-aware access and gateway rate limiting
- [x] Resolve dual Gemini key names deterministically without changing either stored credential
- [ ] Perform human review of high-stakes answers and verifier uncertainty warnings

No repository was pushed and no public deployment was created.
