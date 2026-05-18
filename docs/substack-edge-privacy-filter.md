# Privacy Filters Belong at the Edge

Most agent stacks still treat privacy as something that happens somewhere else.

The app collects text. The agent gets context. Logs get written. Traces get sent. Somewhere downstream, someone promises that sensitive data is being handled carefully.

I do not think that is the right shape for agent infrastructure.

If we are going to put agents into real workflows, especially enterprise workflows, then privacy has to move closer to the edge. It should sit before the model call, before the trace, before the tool invocation, before the data leaves the machine or the network boundary.

That was the reason I built this small harness:

```bash
git clone https://github.com/gupta799/agentci-privacy-filter-harness
cd agentci-privacy-filter-harness
uv sync --dev
```

It runs OpenAI's `openai/privacy-filter` model locally from Hugging Face, using ONNX Runtime on CPU. No hosted inference endpoint. No PyTorch service hiding behind the curtain. No regex fallback pretending to be a model.

Just a local tokenizer, a quantized ONNX artifact, deterministic span decoding, and a FastAPI endpoint that fails closed if the model cannot load.

That last part matters. For privacy infrastructure, a fake fallback is worse than a crash. If the model is not there, I want the service to say so clearly.

## The Thing I Wanted

I wanted a demo that tells a clean story:

1. Download the model from Hugging Face.
2. Run it locally with ONNX Runtime.
3. Redact private spans before they reach an agent.
4. Benchmark the model path with a real inference benchmarking harness.

There are a lot of ways to make a demo look fast. You can benchmark the HTTP endpoint. You can hit it with `ab`, `wrk`, `autocannon`, or `k6`. Those are useful, but they mostly tell you about the API wrapper.

For model deployment, I care about the local inference path:

- How long does the model take to load cold?
- What does p50 latency look like after warmup?
- What happens at p95 and p99?
- How many samples per second can I push through the model?
- Which runtime and provider actually executed the graph?

That is why the repo uses MLPerf Inference LoadGen for the main benchmark. HTTP load testing is still useful as a smoke test, but it is not the headline.

## Running the Local Model

The server starts like this:

```bash
uv run uvicorn agentci_privacy_filter.server:app --host 0.0.0.0 --port 8080
```

On first startup, it downloads only the files needed to run the ONNX model:

```text
config.json
tokenizer.json
tokenizer_config.json
viterbi_calibration.json
onnx/model_quantized.onnx
onnx/model_quantized.onnx_data
```

The default config is intentionally boring:

```bash
MODEL_ID=openai/privacy-filter
ONNX_MODEL_FILE=onnx/model_quantized.onnx
ORT_PROVIDERS=CPUExecutionProvider
ONNX_MAX_TOKENS=4096
```

I like boring here. Boring means I can explain exactly what is running. The model is local. The runtime is ONNX Runtime. The provider is CPU. The decode path is deterministic argmax over BIOES token labels.

You can test the server with:

```bash
curl -s -X POST http://127.0.0.1:8080/redact \
  -H 'content-type: application/json' \
  -d '{"text":"Email Maya Chen at maya.chen@example.com and call +1 415 555 0123."}' | jq
```

The shape of the response is:

```json
{
  "redacted_text": "Email <PRIVATE_PERSON> at <PRIVATE_EMAIL> and call <PRIVATE_PHONE>.",
  "labels": [
    "PRIVATE_EMAIL",
    "PRIVATE_PERSON",
    "PRIVATE_PHONE"
  ],
  "elapsed_ms": 68.5,
  "backend": "onnxruntime",
  "model_id": "openai/privacy-filter",
  "onnx_model_file": "onnx/model_quantized.onnx",
  "decode_mode": "argmax",
  "input_chars": 74,
  "output_chars": 69
}
```

This is the part I want in front of an agent.

Not after the agent has already seen the text. Not after traces have already captured the payload. Before.

## The Core Runtime

The important part is not much code.

```python
from agentci_privacy_filter.redactor import OnnxPrivacyFilterRedactor

redactor = OnnxPrivacyFilterRedactor()
redactor.load()

result = redactor.redact(
    "Email Maya Chen at maya.chen@example.com and call +1 415 555 0123."
)

print(result.redacted_text)
print(result.labels)
```

Under the hood, the redactor does four things:

1. Pulls the model artifacts from Hugging Face with `snapshot_download`.
2. Loads the tokenizer with `tokenizers`.
3. Loads the ONNX graph with `onnxruntime.InferenceSession`.
4. Converts BIOES token predictions into character spans and replaces them with placeholders.

The output placeholders are meant to be stable and boring:

```text
<PRIVATE_EMAIL>
<PRIVATE_PERSON>
<PRIVATE_PHONE>
<SECRET>
```

The point is not to make the text pretty. The point is to make the downstream agent safer to operate.

## Why ONNX Runtime

For edge deployment, I want the runtime to have a few properties:

- It should be local.
- It should be easy to package.
- It should not require a GPU.
- It should make the execution provider visible.
- It should be realistic enough that the benchmark means something.

ONNX Runtime is a good fit for that. It lets me treat the model as an artifact that can be moved around: developer laptop, CI runner, small server, private network, edge box.

The model is still a model, so it should still be evaluated like one. A fast privacy filter that misses obvious private data is not good enough. But speed and deployment shape are separate questions from quality. This repo is about making the runtime path measurable.

## Benchmarking the Model Path

The benchmark command for latency is:

```bash
uv run python scripts/benchmark_loadgen.py \
  --scenario SingleStream \
  --samples examples/benchmark.jsonl \
  --query-count 12 \
  --warmup-samples 3
```

The benchmark command for throughput is:

```bash
uv run python scripts/benchmark_loadgen.py \
  --scenario Offline \
  --samples examples/benchmark.jsonl \
  --query-count 12 \
  --warmup-samples 3
```

The two scenarios are useful for different edge questions.

`SingleStream` is the request path. One input comes in, one answer needs to come back. This is the shape I care about when a user is waiting or an agent is about to call a tool.

`Offline` is the batch path. Give me a pile of local text and tell me how much I can process. This matters for log cleanup, local indexing, corpus scanning, and CI gates.

On my local run, with the quantized ONNX artifact and CPU provider, I saw:

```text
SingleStream
queries: 12
failures: 0
cold model load: 1674 ms
p50 latency: 51.7 ms
p90 latency: 54.1 ms
p95 latency: 54.1 ms
p99 latency: 54.4 ms
throughput: 18.9 samples/sec
```

And for Offline:

```text
Offline
queries: 12
failures: 0
cold model load: 663 ms
p50 latency: 53.4 ms
p90 latency: 54.8 ms
p95 latency: 54.8 ms
p99 latency: 56.8 ms
throughput: 18.2 samples/sec
```

The token throughput on the same tiny fixture corpus was about:

```text
466 tokens/sec
```

That number is not comparable to LLM generation tokens per second. This is token classification throughput: encode text, run the privacy filter, decode spans. Different job, different measurement.

## Why Percentiles Matter

Average latency is usually too comforting.

For an edge privacy filter, I care about the tail. If p50 is fine but p99 is ugly, the system will feel random. Users notice the slow cases. Agents notice them too, because every extra guardrail in the path compounds.

So I look at:

```text
p50: the median request
p90: the slowest 10 percent boundary
p95: the slowest 5 percent boundary
p99: the slowest 1 percent boundary
```

In this case, the spread was tight. That is a good sign for a local CPU path. It means the filter is not just fast on the happy path. It is predictable.

## Where This Fits in an Agent Stack

The practical architecture is simple:

```text
user input
  -> privacy filter
  -> redacted agent context
  -> model/tool call
  -> logs/traces/evals
```

I would use this in a few places:

- Before sending user text to an LLM.
- Before writing traces.
- Before running evals on customer data.
- Before letting an agent call external tools.
- As a CI gate for fixtures, prompts, and test corpora.

That last one is underrated. If your repo contains prompts or conversation fixtures, a privacy filter can be part of the development workflow. Run it before merge. Fail the build if private labels show up where they should not.

In this harness:

```bash
uv run python scripts/privacy_gate.py examples/messages.jsonl
```

The gate exits non-zero when detected labels intersect the configured fail list.

That is the kind of guardrail I like. It is small. It is explicit. It can run locally. It can run in CI. It does not require sending private data somewhere else to find out whether private data exists.

## The Honest Caveat

Speed is not privacy.

This harness proves that the local runtime path works and can be measured. It does not prove recall is good enough for your domain. It does not prove the model catches every secret, every customer identifier, every internal code name, every weirdly formatted phone number.

That has to be evaluated separately.

A serious deployment needs a quality corpus: examples from the real domain, labeled spans, precision, recall, false negatives, false positives, and regression tests. The performance harness tells me whether the system is deployable. The evaluation corpus tells me whether it is trustworthy.

Both matter.

## Why I Built It This Way

I care a lot about the layer between prototype agents and production agents.

That layer is not glamorous. It is mostly boring infrastructure: gates, filters, evals, traces, benchmarks, failure modes, packaging, model artifacts, local runtime choices. But that is exactly where real systems become usable.

The easy demo is to call a hosted model and print a redacted string.

The more interesting demo is to say:

```text
Here is the model artifact.
Here is the runtime.
Here is the endpoint.
Here is the benchmark harness.
Here are the p50/p95/p99 numbers.
Here is where it fails.
Here is how I would put it in front of an agent.
```

That is the kind of agent infrastructure I want more of. Local where it should be local. Measurable where it should be measurable. Honest about what it does and what still needs evaluation.

The repo is here:

https://github.com/gupta799/agentci-privacy-filter-harness

