FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra ml --no-install-project

COPY src ./src
COPY data/samples.jsonl data/few_shot.jsonl ./data/
RUN uv sync --frozen --no-dev --extra ml

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "gstlens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
