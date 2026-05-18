#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mlperf_loadgen as lg

from agentci_privacy_filter.redactor import OnnxPrivacyFilterRedactor


@dataclass(frozen=True)
class BenchmarkSample:
    id: str
    text: str


class LoadGenRunner:
    def __init__(self, samples: list[BenchmarkSample], redactor: OnnxPrivacyFilterRedactor):
        self.samples = samples
        self.redactor = redactor
        self.latencies_ms: list[float] = []
        self.failures = 0
        self.completed = 0

    def load_samples(self, sample_indices: list[int]) -> None:
        del sample_indices

    def unload_samples(self, sample_indices: list[int]) -> None:
        del sample_indices

    def issue_queries(self, query_samples: list[lg.QuerySample]) -> None:
        responses = []
        for sample in query_samples:
            started = time.perf_counter()
            try:
                self.redactor.redact(self.samples[sample.index].text)
            except Exception:
                self.failures += 1
            finally:
                self.completed += 1
                self.latencies_ms.append((time.perf_counter() - started) * 1000)
                responses.append(lg.QuerySampleResponse(sample.id, 0, 0))
        lg.QuerySamplesComplete(responses)

    def flush_queries(self) -> None:
        return None


def main() -> int:
    args = parse_args()
    samples = load_samples(args.samples)
    redactor = OnnxPrivacyFilterRedactor()

    cold_started = time.perf_counter()
    model_info = redactor.load()
    cold_model_load_ms = (time.perf_counter() - cold_started) * 1000

    for sample in samples[: args.warmup_samples]:
        redactor.redact(sample.text)

    runner = LoadGenRunner(samples, redactor)
    settings = lg.TestSettings()
    settings.scenario = scenario_from_name(args.scenario)
    settings.mode = lg.TestMode.PerformanceOnly
    settings.min_query_count = args.query_count
    settings.max_query_count = args.query_count
    settings.min_duration_ms = 0
    settings.performance_sample_count_override = len(samples)
    settings.print_timestamps = False
    if settings.scenario == lg.TestScenario.Offline:
        settings.offline_expected_qps = args.offline_expected_qps
    else:
        settings.single_stream_expected_latency_ns = args.single_stream_expected_latency_ms * 1_000_000

    sut = lg.ConstructSUT(runner.issue_queries, runner.flush_queries)
    qsl = lg.ConstructQSL(
        len(samples),
        len(samples),
        runner.load_samples,
        runner.unload_samples,
    )
    started = time.perf_counter()
    try:
        lg.StartTest(sut, qsl, settings)
    finally:
        lg.DestroyQSL(qsl)
        lg.DestroySUT(sut)
    total_seconds = time.perf_counter() - started

    summary = build_summary(
        args=args,
        model_info=model_info,
        cold_model_load_ms=cold_model_load_ms,
        runner=runner,
        total_seconds=total_seconds,
    )
    print(json.dumps(summary, indent=2))
    return 1 if runner.failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLPerf LoadGen edge benchmark.")
    parser.add_argument(
        "--scenario",
        choices=["SingleStream", "Offline"],
        default="SingleStream",
    )
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=20)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--offline-expected-qps", type=float, default=1.0)
    parser.add_argument("--single-stream-expected-latency-ms", type=float, default=1000.0)
    return parser.parse_args()


def load_samples(path: Path) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        samples.append(BenchmarkSample(id=str(payload["id"]), text=str(payload["text"])))
    if not samples:
        raise ValueError(f"No benchmark samples found in {path}")
    return samples


def scenario_from_name(name: str):
    if name == "SingleStream":
        return lg.TestScenario.SingleStream
    if name == "Offline":
        return lg.TestScenario.Offline
    raise ValueError(f"Unsupported scenario: {name}")


def build_summary(
    *,
    args: argparse.Namespace,
    model_info,
    cold_model_load_ms: float,
    runner: LoadGenRunner,
    total_seconds: float,
) -> dict[str, object]:
    latencies = runner.latencies_ms
    return {
        "scenario": args.scenario,
        "samples": len(runner.samples),
        "queries": runner.completed,
        "failures": runner.failures,
        "total_seconds": total_seconds,
        "samples_per_second": runner.completed / total_seconds if total_seconds else 0,
        "cold_model_load_ms": cold_model_load_ms,
        "latency_ms": percentile_summary(latencies),
        "model": {
            "model_id": model_info.model_id,
            "onnx_model_file": model_info.onnx_model_file,
            "model_bytes": model_info.model_bytes,
            "providers": model_info.providers,
        },
    }


def percentile_summary(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "mean": 0}
    return {
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "mean": statistics.fmean(ordered),
    }


def percentile(ordered: list[float], fraction: float) -> float:
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


if __name__ == "__main__":
    raise SystemExit(main())
