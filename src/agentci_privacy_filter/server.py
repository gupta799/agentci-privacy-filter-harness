from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentci_privacy_filter.redactor import RedactorError, build_redactor


app = FastAPI(title="AgentCI Privacy Filter Harness", version="0.1.0")
redactor = build_redactor()


class RedactRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000_000)


class RedactResponse(BaseModel):
    redacted_text: str
    labels: list[str]
    elapsed_ms: float
    backend: str
    input_chars: int
    output_chars: int


@app.get("/healthz")
def healthz():
    return {"ok": True, "backend": redactor.__class__.__name__}


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
        input_chars=len(request.text),
        output_chars=len(result.redacted_text),
    )
