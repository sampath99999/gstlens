from typing import Any, Literal

from pydantic import BaseModel, Field

from gstlens.extraction.base import Variant
from gstlens.schema import Invoice


class ExtractRequest(BaseModel):
    text: str = Field(min_length=20, description="Plain text of the invoice")
    model: Variant = "finetuned"


class CheckOut(BaseModel):
    name: str
    passed: bool
    detail: str


class ExtractResponse(BaseModel):
    model: Variant
    invoice: Invoice
    checks: list[CheckOut]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    json_repaired: bool
    raw_output: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)
    raw_output: str | None = None
    latency_ms: float | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"]
    backend: str
    base_model: str
    models: list[Variant]


class SampleOut(BaseModel):
    id: str
    template: str
    text: str
