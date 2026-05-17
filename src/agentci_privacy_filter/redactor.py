from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass


PLACEHOLDER_RE = re.compile(r"\[(PRIVATE_[A-Z_]+|ACCOUNT_NUMBER|SECRET)\]")


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    labels: list[str]
    elapsed_ms: float
    backend: str


class RedactorError(RuntimeError):
    pass


class OpfRedactor:
    def __init__(self, command: str = "opf", device: str = "cpu", timeout_seconds: int = 120):
        self.command = command
        self.device = device
        self.timeout_seconds = timeout_seconds

    def redact(self, text: str) -> RedactionResult:
        start = time.perf_counter()
        command = [*shlex.split(self.command), "--device", self.device, text]
        env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb"}

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                env=env,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RedactorError(
                "Could not find the `opf` CLI. Install OpenAI Privacy Filter first."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RedactorError("OpenAI Privacy Filter timed out.") from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RedactorError(f"OpenAI Privacy Filter failed: {stderr}")

        redacted_text = completed.stdout.strip()
        return RedactionResult(
            redacted_text=redacted_text,
            labels=extract_labels(redacted_text),
            elapsed_ms=elapsed_ms,
            backend="opf",
        )


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
            command=os.getenv("OPF_COMMAND", "opf"),
            device=os.getenv("OPF_DEVICE", "cpu"),
            timeout_seconds=int(os.getenv("OPF_TIMEOUT_SECONDS", "120")),
        )
    raise RedactorError(f"Unknown REDACTOR_BACKEND={backend!r}")
