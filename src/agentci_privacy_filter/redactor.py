from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass


PLACEHOLDER_RE = re.compile(r"[\[<]([A-Z0-9_]+)[>\]]")


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    labels: list[str]
    elapsed_ms: float
    backend: str


class RedactorError(RuntimeError):
    pass


class OpfRedactor:
    def __init__(self, device: str = "cpu", output_mode: str = "typed"):
        self.device = device
        self.output_mode = output_mode
        self._redactor = None

    def redact(self, text: str) -> RedactionResult:
        start = time.perf_counter()
        redactor = self._get_redactor()

        try:
            result = redactor.redact(text)
        except Exception as exc:
            raise RedactorError(f"OpenAI Privacy Filter failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        labels = sorted({span.label.upper() for span in result.detected_spans})
        return RedactionResult(
            redacted_text=result.redacted_text,
            labels=labels,
            elapsed_ms=elapsed_ms,
            backend="opf",
        )

    def _get_redactor(self):
        if self._redactor is not None:
            return self._redactor

        try:
            from opf import OPF
        except ImportError as exc:
            raise RedactorError(
                "OpenAI Privacy Filter is not installed. Run `uv sync --dev --extra opf`."
            ) from exc

        self._redactor = OPF(device=self.device, output_mode=self.output_mode)
        return self._redactor


class RegexRedactor:
    """Fast local smoke-test backend. Do not use this for model benchmarks."""

    EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    PHONE_RE = re.compile(r"\+?\d[\d .()/-]{7,}\d")
    SECRET_RE = re.compile(r"\b(?:sk|pk|ghp|gho|xoxb|AKIA)[A-Za-z0-9_-]{8,}\b")

    def redact(self, text: str) -> RedactionResult:
        start = time.perf_counter()
        redacted = self.EMAIL_RE.sub("[PRIVATE_EMAIL]", text)
        redacted = self.SECRET_RE.sub("[SECRET]", redacted)
        redacted = self.PHONE_RE.sub("[PRIVATE_PHONE]", redacted)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RedactionResult(
            redacted_text=redacted,
            labels=extract_labels(redacted),
            elapsed_ms=elapsed_ms,
            backend="regex",
        )


def extract_labels(text: str) -> list[str]:
    return sorted(set(PLACEHOLDER_RE.findall(text)))


def build_redactor():
    backend = os.getenv("REDACTOR_BACKEND", "opf").lower()
    if backend == "regex":
        return RegexRedactor()
    if backend == "opf":
        return OpfRedactor(
            device=os.getenv("OPF_DEVICE", "cpu"),
            output_mode=os.getenv("OPF_OUTPUT_MODE", "typed"),
        )
    raise RedactorError(f"Unknown REDACTOR_BACKEND={backend!r}")
