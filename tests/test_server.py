from fastapi.testclient import TestClient

from agentci_privacy_filter import server
from agentci_privacy_filter.redactor import ModelInfo, RedactionResult


class FakeRedactor:
    backend = "onnxruntime"
    decode_mode = "argmax"
    ready = False

    def load(self):
        self.ready = True
        return self.info()

    def info(self):
        return ModelInfo(
            model_id="openai/privacy-filter",
            onnx_model_file="onnx/model_quantized.onnx",
            providers=["CPUExecutionProvider"],
            model_bytes=123,
            model_path="/tmp/model.onnx",
        )

    def redact(self, text: str):
        return RedactionResult(
            redacted_text="Email <PRIVATE_PERSON> at <PRIVATE_EMAIL>.",
            labels=["PRIVATE_EMAIL", "PRIVATE_PERSON"],
            elapsed_ms=1.5,
            backend=self.backend,
            model_id="openai/privacy-filter",
            onnx_model_file="onnx/model_quantized.onnx",
            decode_mode=self.decode_mode,
        )


def test_healthz_reports_onnx_backend(monkeypatch):
    monkeypatch.setattr(server, "redactor", FakeRedactor())
    monkeypatch.setattr(server, "model_info", None)

    with TestClient(server.app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["backend"] == "onnxruntime"
    assert response.json()["ready"] is True


def test_redact_response_includes_model_metadata(monkeypatch):
    monkeypatch.setattr(server, "redactor", FakeRedactor())
    monkeypatch.setattr(server, "model_info", None)

    with TestClient(server.app) as client:
        response = client.post("/redact", json={"text": "Email Maya at maya@example.com."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "onnxruntime"
    assert payload["model_id"] == "openai/privacy-filter"
    assert payload["onnx_model_file"] == "onnx/model_quantized.onnx"
    assert payload["redacted_text"] == "Email <PRIVATE_PERSON> at <PRIVATE_EMAIL>."
