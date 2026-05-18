from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentci_privacy_filter.redactor import ModelInfo, RedactorError, build_redactor


redactor = build_redactor()
model_info: ModelInfo | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    global model_info
    model_info = redactor.load()
    yield


app = FastAPI(
    title="AgentCI Privacy Filter Harness",
    version="0.1.0",
    lifespan=lifespan,
)


class RedactRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000_000)


class RedactResponse(BaseModel):
    redacted_text: str
    labels: list[str]
    elapsed_ms: float
    backend: str
    model_id: str
    onnx_model_file: str
    decode_mode: str
    input_chars: int
    output_chars: int


@app.get("/healthz")
def healthz():
    info = model_info or redactor.info()
    return {
        "ok": redactor.ready,
        "backend": redactor.backend,
        "ready": redactor.ready,
        "model_id": info.model_id,
        "onnx_model_file": info.onnx_model_file,
        "providers": info.providers,
        "model_bytes": info.model_bytes,
        "decode_mode": redactor.decode_mode,
    }


@app.post("/redact")
def redact(request: RedactRequest) -> RedactResponse:
    try:
        result = redactor.redact(request.text)
    except RedactorError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return RedactResponse(
        redacted_text=result.redacted_text,
        labels=result.labels,
        elapsed_ms=result.elapsed_ms,
        backend=result.backend,
        model_id=result.model_id,
        onnx_model_file=result.onnx_model_file,
        decode_mode=result.decode_mode,
        input_chars=len(request.text),
        output_chars=len(result.redacted_text),
    )
