import pytest

from agentci_privacy_filter import redactor
from agentci_privacy_filter.redactor import (
    DetectedSpan,
    RedactorError,
    apply_redactions,
    decode_bioes_spans,
)


def test_bioes_decoder_handles_multi_and_single_token_spans():
    text = "Email Maya Chen at maya@example.com today"
    labels = [
        "O",
        "B-private_person",
        "E-private_person",
        "O",
        "S-private_email",
        "O",
    ]
    offsets = [(0, 5), (5, 10), (10, 15), (15, 18), (19, 35), (36, 41)]

    spans = decode_bioes_spans(text, labels, offsets)

    assert spans == [
        DetectedSpan(
            label="private_person",
            start=6,
            end=15,
            text="Maya Chen",
            placeholder="<PRIVATE_PERSON>",
        ),
        DetectedSpan(
            label="private_email",
            start=19,
            end=35,
            text="maya@example.com",
            placeholder="<PRIVATE_EMAIL>",
        ),
    ]


def test_apply_redactions_preserves_surrounding_text():
    text = "Email Maya Chen at maya@example.com."
    spans = [
        DetectedSpan("private_person", 6, 15, "Maya Chen", "<PRIVATE_PERSON>"),
        DetectedSpan("private_email", 19, 35, "maya@example.com", "<PRIVATE_EMAIL>"),
    ]

    assert (
        apply_redactions(text, spans)
        == "Email <PRIVATE_PERSON> at <PRIVATE_EMAIL>."
    )


def test_missing_model_file_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(redactor, "snapshot_download", lambda **_: str(tmp_path))
    (tmp_path / "config.json").write_text('{"id2label": {"0": "O"}}')
    (tmp_path / "tokenizer.json").write_text("{}")
    backend = redactor.OnnxPrivacyFilterRedactor(onnx_model_file="missing.onnx")

    with pytest.raises(RedactorError, match="Missing ONNX model file"):
        backend.load()


def test_only_onnx_backend_is_constructed():
    assert isinstance(redactor.build_redactor(), redactor.OnnxPrivacyFilterRedactor)
