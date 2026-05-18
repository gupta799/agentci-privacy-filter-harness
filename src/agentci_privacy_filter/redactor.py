from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer


DEFAULT_MODEL_ID = "openai/privacy-filter"
DEFAULT_ONNX_MODEL_FILE = "onnx/model_quantized.onnx"
DEFAULT_ALLOW_PATTERNS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "viterbi_calibration.json",
    "onnx/model_quantized.onnx",
    "onnx/model_quantized.onnx_data",
)


@dataclass(frozen=True)
class DetectedSpan:
    label: str
    start: int
    end: int
    text: str
    placeholder: str


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    labels: list[str]
    elapsed_ms: float
    backend: str
    model_id: str
    onnx_model_file: str
    decode_mode: str


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    onnx_model_file: str
    providers: list[str]
    model_bytes: int
    model_path: str


class RedactorError(RuntimeError):
    pass


class OnnxPrivacyFilterRedactor:
    backend = "onnxruntime"
    decode_mode = "argmax"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        onnx_model_file: str = DEFAULT_ONNX_MODEL_FILE,
        providers: list[str] | None = None,
        allow_patterns: tuple[str, ...] = DEFAULT_ALLOW_PATTERNS,
        max_tokens: int = 4096,
    ) -> None:
        self.model_id = model_id
        self.onnx_model_file = onnx_model_file
        self.providers = providers or ["CPUExecutionProvider"]
        self.allow_patterns = allow_patterns
        self.max_tokens = max_tokens
        self._model_dir: Path | None = None
        self._session: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._id_to_label: dict[int, str] = {}
        self._model_bytes = 0

    @property
    def ready(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    def load(self) -> ModelInfo:
        if self.ready:
            return self.info()

        try:
            model_dir = Path(
                snapshot_download(
                    repo_id=self.model_id,
                    allow_patterns=list(self.allow_patterns),
                )
            )
            config = json.loads((model_dir / "config.json").read_text())
            self._id_to_label = {
                int(index): label for index, label in config["id2label"].items()
            }
            model_path = model_dir / self.onnx_model_file
            if not model_path.exists():
                raise FileNotFoundError(f"Missing ONNX model file: {model_path}")
            self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
            self._session = ort.InferenceSession(str(model_path), providers=self.providers)
            self._model_dir = model_dir
            self._model_bytes = _artifact_size(model_path)
        except Exception as exc:
            raise RedactorError(f"Failed to load ONNX privacy model: {exc}") from exc

        return self.info()

    def info(self) -> ModelInfo:
        if self._session is None or self._model_dir is None:
            return ModelInfo(
                model_id=self.model_id,
                onnx_model_file=self.onnx_model_file,
                providers=[],
                model_bytes=0,
                model_path="",
            )
        return ModelInfo(
            model_id=self.model_id,
            onnx_model_file=self.onnx_model_file,
            providers=self._session.get_providers(),
            model_bytes=self._model_bytes,
            model_path=str(self._model_dir / self.onnx_model_file),
        )

    def redact(self, text: str) -> RedactionResult:
        start = time.perf_counter()
        self.load()
        assert self._session is not None
        assert self._tokenizer is not None

        encoding = self._tokenizer.encode(text)
        ids = encoding.ids[: self.max_tokens]
        offsets = encoding.offsets[: self.max_tokens]
        if not ids:
            return self._result(text, [], start)

        input_ids = np.array([ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        logits = self._session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0][0]
        labels = [self._id_to_label[int(index)] for index in logits.argmax(axis=-1)]
        spans = decode_bioes_spans(text, labels, offsets)
        return self._result(text, spans, start)

    def _result(
        self, text: str, spans: list[DetectedSpan], start: float
    ) -> RedactionResult:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RedactionResult(
            redacted_text=apply_redactions(text, spans),
            labels=sorted({span.placeholder.strip("<>") for span in spans}),
            elapsed_ms=elapsed_ms,
            backend=self.backend,
            model_id=self.model_id,
            onnx_model_file=self.onnx_model_file,
            decode_mode=self.decode_mode,
        )


def build_redactor() -> OnnxPrivacyFilterRedactor:
    model_file = os.getenv("ONNX_MODEL_FILE", DEFAULT_ONNX_MODEL_FILE)
    allow_patterns = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "viterbi_calibration.json",
        model_file,
        f"{model_file}_data",
    )
    return OnnxPrivacyFilterRedactor(
        model_id=os.getenv("MODEL_ID", DEFAULT_MODEL_ID),
        onnx_model_file=model_file,
        providers=os.getenv("ORT_PROVIDERS", "CPUExecutionProvider").split(","),
        allow_patterns=allow_patterns,
        max_tokens=int(os.getenv("ONNX_MAX_TOKENS", "4096")),
    )


def decode_bioes_spans(
    text: str, labels: list[str], offsets: list[tuple[int, int]]
) -> list[DetectedSpan]:
    spans: list[DetectedSpan] = []
    active_label: str | None = None
    active_start: int | None = None
    active_end: int | None = None

    def flush() -> None:
        nonlocal active_label, active_start, active_end
        if active_label is not None and active_start is not None and active_end is not None:
            spans.append(_span_from_offsets(text, active_label, active_start, active_end))
        active_label = None
        active_start = None
        active_end = None

    for raw_label, (start, end) in zip(labels, offsets, strict=False):
        if raw_label == "O" or end <= start:
            flush()
            continue
        prefix, label = _split_bioes(raw_label)
        if prefix == "S":
            flush()
            spans.append(_span_from_offsets(text, label, start, end))
        elif prefix == "B":
            flush()
            active_label = label
            active_start = start
            active_end = end
        elif prefix == "I":
            if active_label == label and active_start is not None:
                active_end = end
            else:
                flush()
                active_label = label
                active_start = start
                active_end = end
        elif prefix == "E":
            if active_label == label and active_start is not None:
                active_end = end
                flush()
            else:
                flush()
                spans.append(_span_from_offsets(text, label, start, end))
        else:
            flush()
    flush()
    return _select_non_overlapping(spans)


def apply_redactions(text: str, spans: list[DetectedSpan]) -> str:
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for span in _select_non_overlapping(spans):
        pieces.append(text[cursor : span.start])
        pieces.append(span.placeholder)
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _split_bioes(label: str) -> tuple[str, str]:
    if "-" not in label:
        return label, label
    prefix, value = label.split("-", 1)
    return prefix, value


def _span_from_offsets(text: str, label: str, start: int, end: int) -> DetectedSpan:
    start, end = _trim_span(text, start, end)
    return DetectedSpan(
        label=label,
        start=start,
        end=end,
        text=text[start:end],
        placeholder=f"<{label.upper()}>",
    )


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _select_non_overlapping(spans: list[DetectedSpan]) -> list[DetectedSpan]:
    selected: list[DetectedSpan] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: (item.start, -(item.end - item.start))):
        if span.start < cursor or span.end <= span.start:
            continue
        selected.append(span)
        cursor = span.end
    return selected


def _artifact_size(model_path: Path) -> int:
    total = model_path.stat().st_size
    data_path = Path(f"{model_path}_data")
    if data_path.exists():
        total += data_path.stat().st_size
    return total
