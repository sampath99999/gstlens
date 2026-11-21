.PHONY: install data train train-smoke eval report api web test lint

install:
	uv sync --all-extras
	cd web && npm install

data:
	uv run gstlens-data

train-smoke:
	uv run gstlens-train --limit 48 --max-steps 6 --grad-accum 4 --eval-steps 6 --eval-examples 8 --output-dir artifacts/smoke

train:
	uv run gstlens-train

eval:
	uv run gstlens-eval run --systems base-zero base-few finetuned

report:
	uv run gstlens-eval report

api:
	uv run uvicorn gstlens.api.main:app --port 8000

web:
	cd web && npm run dev

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
