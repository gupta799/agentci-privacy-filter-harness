# AgentCI Privacy Filter Harness

Run an on-prem privacy model as an agent CI gate, then test it like production
infrastructure.

This repo is a companion scaffold for a post about using local models at agent
boundaries. The core idea is simple: before an agent reads logs, tickets, email,
browser output, or tool traces, route the text through a local privacy filter and
fail the run when sensitive spans appear.

The real backend is OpenAI Privacy Filter via the `opf` Python API, so the
model stays warm in the server process. A small regex backend exists only for
smoke tests when the model is not installed yet.

## Quick Start

From a checkout of this repo:

```bash
uv sync --dev
```

Install the real OpenAI Privacy Filter backend:

```bash
uv sync --dev --extra opf
```

Run the server:

```bash
REDACTOR_BACKEND=opf OPF_DEVICE=cpu uv run uvicorn agentci_privacy_filter.server:app --host 0.0.0.0 --port 8080
```

Try it:

```bash
curl -s http://localhost:8080/redact \
  -H 'content-type: application/json' \
  -d '{"text":"Email Maya Chen at maya@example.com and use key sk-test-123."}' | jq
```

## AgentCI Gate

Fail a run if files contain labels that should not reach an agent:

```bash
uv run python scripts/privacy_gate.py examples/messages.jsonl
```

The gate exits non-zero when the redacted output contains labels such as
`PRIVATE_EMAIL`, `PRIVATE_PHONE`, `ACCOUNT_NUMBER`, or `SECRET`.

## Load Testing

With the server running:

```bash
k6 run load-tests/k6-redact.js
```

Or with autocannon:

```bash
npm install
node load-tests/autocannon-redact.mjs
```

Useful numbers for the post:

- requests per second
- p50, p95, p99 latency
- error rate under load
- CPU and memory for the redaction server
- cold start time for the model

## Why This Matters

On-prem models are most useful when they are narrow, measurable, and placed at
system boundaries. Privacy filtering is a perfect example: it is not a chatbot,
it is a gate. The test is not whether it sounds smart; the test is whether it
protects sensitive text before an agent sees it, and whether it holds up under
real traffic.
