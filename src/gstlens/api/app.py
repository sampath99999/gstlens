import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from gstlens.api.models import (
    CheckOut,
    ErrorBody,
    ErrorResponse,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    SampleOut,
)
from gstlens.config import Settings, get_settings
from gstlens.data.build import read_jsonl
from gstlens.extraction.service import InvoiceExtractor

log = logging.getLogger(__name__)

ExtractorFactory = Callable[[Settings], InvoiceExtractor]


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.body = ErrorBody(code=code, message=message, **extra)


def build_extractor(settings: Settings) -> InvoiceExtractor:
    few_shot = read_jsonl(settings.few_shot_path) if settings.base_prompt == "few_shot" else []
    if settings.backend == "openai":
        from gstlens.extraction.openai_compat import OpenAICompatGenerator

        generator = OpenAICompatGenerator(
            settings.openai_base_url, settings.openai_api_key, settings.openai_model
        )
        return InvoiceExtractor(generator, {"base": few_shot})

    from gstlens.extraction.hf import HFGenerator

    adapter = settings.adapter_path
    if adapter is not None and not adapter.exists():
        log.warning("adapter %s not found, serving the base model only", adapter)
        adapter = None
    hf = HFGenerator(
        settings.base_model,
        adapter,
        settings.device,
        settings.dtype,
        settings.max_new_tokens,
    )
    return InvoiceExtractor(hf, {"base": few_shot})


def create_app(
    settings: Settings | None = None, extractor_factory: ExtractorFactory = build_extractor
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.extractor = await run_in_threadpool(extractor_factory, settings)
        app.state.samples = (
            read_jsonl(settings.samples_path) if settings.samples_path.exists() else []
        )
        log.info("extractor ready, variants=%s", app.state.extractor.variants)
        yield

    app = FastAPI(title="GSTLens", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"error": exc.body.model_dump()})

    @app.exception_handler(RequestValidationError)
    async def _request_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
        ]
        body = ErrorBody(
            code="invalid_request", message="Request validation failed", details=details
        )
        return JSONResponse(status_code=422, content={"error": body.model_dump()})

    def get_extractor(request: Request) -> InvoiceExtractor:
        return request.app.state.extractor

    Extractor = Annotated[InvoiceExtractor, Depends(get_extractor)]

    @app.get("/health", response_model=HealthResponse)
    async def health(extractor: Extractor) -> HealthResponse:
        return HealthResponse(
            status="ok",
            backend=settings.backend,
            base_model=settings.base_model,
            models=list(extractor.variants),
        )

    @app.get("/samples", response_model=list[SampleOut])
    async def samples(request: Request) -> list[SampleOut]:
        return [
            SampleOut(id=s["id"], template=s["template"], text=s["text"])
            for s in request.app.state.samples
        ]

    @app.post(
        "/extract",
        response_model=ExtractResponse,
        responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    async def extract(body: ExtractRequest, extractor: Extractor) -> ExtractResponse:
        if len(body.text) > settings.max_input_chars:
            raise ApiError(
                413, "text_too_long", f"Invoice text is over {settings.max_input_chars} characters"
            )
        if body.model not in extractor.variants:
            raise ApiError(
                400,
                "model_unavailable",
                f"Model '{body.model}' is not loaded",
                details=[{"available": list(extractor.variants)}],
            )
        result = await run_in_threadpool(extractor.extract, body.text, body.model)
        gen = result.generation
        if result.invoice is None:
            raise ApiError(
                422,
                "invalid_model_output",
                "The model output did not match the invoice schema",
                details=result.errors,
                raw_output=gen.text,
                latency_ms=round(gen.latency_s * 1000, 1),
            )
        return ExtractResponse(
            model=body.model,
            invoice=result.invoice,
            checks=[CheckOut(name=c.name, passed=c.passed, detail=c.detail) for c in result.checks],
            latency_ms=round(gen.latency_s * 1000, 1),
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            json_repaired=result.parsed.repaired,
            raw_output=gen.text,
        )

    return app
