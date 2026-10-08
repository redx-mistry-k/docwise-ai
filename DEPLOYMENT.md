# DocWise AI deployment readiness

This document is a deployment assessment and checklist. It does not create or publish a service.

## Recommended portfolio path

For a durable, password-protected recruiter demo, use the existing Docker image on a single
container web service with a persistent disk mounted at `/app/storage`. Set
`DOCWISE_STORAGE_DIR=/app/storage`, keep one instance, and put identity-aware access control in
front of the service. Render is a straightforward fit for this topology: its Docker web services
support health checks and its paid persistent disks preserve files across restarts. Render states
that the default service filesystem is ephemeral and that a disk is attached to only one service
instance, which matches DocWise's current single-instance registry design:

- https://render.com/docs/your-first-deploy
- https://render.com/docs/disks

Streamlit Community Cloud is the simplest disposable showcase. It supports secret management and
private-app access, but it should be treated as ephemeral for DocWise's uploaded files, Chroma
index, registry, and SQLite quota ledger. Use it only with synthetic showcase documents and accept
that an app restart can require re-indexing:

- https://docs.streamlit.io/deploy/concepts/secrets
- https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/trust-and-security

Fly.io or another container host with an attached persistent volume is also technically suitable.
The critical requirement is not the brand of host: it is one application instance, durable local
storage, HTTPS, protected access, bounded resource use, and outbound HTTPS access to Gemini and the
initial embedding-model download.

## Hosting tradeoffs

| Option | Good fit | Important constraint |
|---|---|---|
| Streamlit Community Cloud | Fast, disposable recruiter demo using synthetic documents | Treat runtime storage as ephemeral; do not upload confidential material |
| Render web service + disk | Recommended durable single-instance demo using the existing Dockerfile | Persistent disks are paid and prevent horizontal scaling of this architecture |
| Fly.io/VM + volume | Greater infrastructure control | You own more TLS, access, backup, and operations work |
| Multi-instance production service | Not ready without architectural work | Replace the JSON registry/local Chroma/SQLite coordination with transactional managed services |

## Required secrets and settings

Configure secrets in the hosting platform; never commit them.

```dotenv
GOOGLE_API_KEY=<platform-secret>
DOCWISE_ACCESS_PASSWORD=<long-random-platform-secret>
DOCWISE_QUERY_RPM=10
DOCWISE_STORAGE_DIR=/app/storage
GEMINI_ROUTING_MODE=balanced
```

`GEMINI_API_KEY` remains a compatibility alias. Prefer `GOOGLE_API_KEY`. When both exist, DocWise
uses `GOOGLE_API_KEY` explicitly and does not overwrite or delete either value.

## Pre-deployment checklist

- Use only synthetic or approved non-confidential documents in a public portfolio demo.
- Keep the service at one instance while it uses the local JSON registry, Chroma, and SQLite.
- Mount durable storage at `/app/storage` if uploads must survive restarts.
- Configure the API key and access password through the platform's secret manager.
- Place identity-aware authentication and gateway rate limiting in front of the built-in controls
  for anything beyond a low-risk portfolio demo.
- Confirm HTTPS, request-size limits, logs, backups, dependency patching, and cost alerts.
- Pre-download or allow the first-start download of `all-MiniLM-L6-v2`.
- Run the unit/integration suite, opt-in browser suite, image build, container health check, and a
  synthetic upload/chat/delete smoke test in the target environment.
- Review Gemini's current free-tier availability and quotas in the actual Google project. Local
  quota tracking is conservative but is not authoritative billing protection.

## Current deployment limitations

- The built-in password is a shared gate, not user identity or authorization.
- The request limiter is per Streamlit session and can be bypassed with new sessions.
- Runtime documents are not encrypted by DocWise itself; rely on encrypted host storage and access
  controls, and avoid sensitive uploads in a showcase.
- There is no malware scanning or OCR pipeline.
- Local storage and process-cached state prevent safe horizontal scaling.
- A successful free-tier Gemini call does not guarantee future model availability or zero cost.
