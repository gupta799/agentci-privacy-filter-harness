# AgentCI Privacy Filter Harness

Run OpenAI Privacy Filter as an on-prem ONNX Runtime model, put it in front of
agent inputs, and benchmark it with an edge-style MLPerf LoadGen harness.

This repo is intentionally strict: there is no regex fallback and no dependency
on OpenAI's Python/PyTorch repo. The server downloads `openai/privacy-filter`
from Hugging Face, loads the quantized ONNX artifact locally, and fails if the
real model cannot run.

## Quick Start

From a checkout of this repo:

```bash
uv sync --dev
```

Run the ONNX Runtime server:

```bash
uv run uvicorn agentci_privacy_filter.server:app --host 0.0.0.0 --port 8080
```

First startup downloads only the required Hugging Face files:

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `viterbi_calibration.json`
- `onnx/model_quantized.onnx`
- `onnx/model_quantized.onnx_data`

Try it:

```bash
curl -s http://localhost:8080/redact \
  -H 'content-type: application/json' \
  -d '{"text":"Email Maya Chen at maya@example.com and call +1 415 555 0123."}' | jq
```

## Runtime Configuration

Defaults:

```bash
MODEL_ID=openai/privacy-filter
ONNX_MODEL_FILE=onnx/model_quantized.onnx
ORT_PROVIDERS=CPUExecutionProvider
ONNX_MAX_TOKENS=4096
```

The default decoder is deterministic BIOES argmax. It does not use OpenAI's
Viterbi calibration path, so benchmark reports identify `decode_mode="argmax"`.

## API

Health:

```bash
curl -s http://localhost:8080/healthz | jq
```

Redact:

```bash
curl -s http://localhost:8080/redact \
  -H 'content-type: application/json' \
  -d '{"text":"Email Maya at maya@example.com"}' | jq
```

`POST /redact` returns:

- `redacted_text`
- `labels`
- `elapsed_ms`
- `backend`
- `model_id`
- `onnx_model_file`
- `decode_mode`
- `input_chars`
- `output_chars`

## AgentCI Gate

Fail a run if files contain labels that should not reach an agent:

```bash
uv run python scripts/privacy_gate.py examples/messages.jsonl
```

The gate exits non-zero when detected labels intersect the configured fail list.

## Edge Benchmarking

The canonical model benchmark uses MLPerf Inference LoadGen:

```bash
uv run python scripts/benchmark_loadgen.py \
  --scenario SingleStream \
  --samples examples/benchmark.jsonl
```

```bash
uv run python scripts/benchmark_loadgen.py \
  --scenario Offline \
  --samples examples/benchmark.jsonl
```

The benchmark reports:

- cold model load time
- p50/p90/p95/p99 latency
- samples/sec
- failures
- model artifact name and size
- ONNX Runtime providers

`k6` is kept only as optional API smoke testing for the HTTP layer:

```bash
k6 run load-tests/k6-redact.js
```

Speed is not enough for a privacy filter. Pair these benchmark numbers with a
quality evaluation corpus that measures precision, recall, and false negatives.
