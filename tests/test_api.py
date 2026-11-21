import json

import pytest
from fastapi.testclient import TestClient

from gstlens.api.app import create_app
from gstlens.config import Settings
from gstlens.extraction.service import InvoiceExtractor
from tests.conftest import FakeGenerator

FENCE = "`" * 3


@pytest.fixture
def fake(target_json: str) -> FakeGenerator:
    return FakeGenerator({"finetuned": target_json, "base": f"{FENCE}json\n{target_json}\n{FENCE}"})


@pytest.fixture
def client(fake: FakeGenerator, tmp_path):
    few_shot = [{"text": "example invoice", "target": {"a": 1}}]
    settings = Settings(samples_path=tmp_path / "missing.jsonl", max_input_chars=5000)
    app = create_app(settings, lambda _: InvoiceExtractor(fake, {"base": few_shot}))
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["models"] == ["base", "finetuned"]


def test_extract_finetuned(client: TestClient, sample_record: dict, fake: FakeGenerator) -> None:
    res = client.post("/extract", json={"text": sample_record["text"], "model": "finetuned"})
    assert res.status_code == 200
    body = res.json()
    assert body["invoice"] == sample_record["target"]
    assert body["json_repaired"] is False
    assert body["latency_ms"] == 250.0
    assert all(c["passed"] for c in body["checks"])
    messages, variant = fake.calls[-1]
    assert variant == "finetuned"
    assert len(messages) == 2


def test_base_model_gets_few_shot_and_repair(
    client: TestClient, sample_record: dict, fake: FakeGenerator
) -> None:
    res = client.post("/extract", json={"text": sample_record["text"], "model": "base"})
    assert res.status_code == 200
    assert res.json()["json_repaired"] is True
    messages, _ = fake.calls[-1]
    assert len(messages) == 4


def test_schema_violation_returns_structured_error(
    client: TestClient, fake: FakeGenerator, sample_record: dict
) -> None:
    broken = dict(sample_record["target"], seller_gstin="NOT-A-GSTIN")
    fake.replies["finetuned"] = json.dumps(broken)
    res = client.post("/extract", json={"text": sample_record["text"]})
    assert res.status_code == 422
    error = res.json()["error"]
    assert error["code"] == "invalid_model_output"
    assert error["details"][0]["loc"] == ["seller_gstin"]
    assert "NOT-A-GSTIN" in error["raw_output"]


def test_unparseable_output(client: TestClient, fake: FakeGenerator, sample_record: dict) -> None:
    fake.replies["finetuned"] = "sorry"
    res = client.post("/extract", json={"text": sample_record["text"]})
    assert res.status_code == 422
    assert res.json()["error"]["details"][0]["type"] == "json_invalid"


def test_failed_consistency_check_is_reported(
    client: TestClient, fake: FakeGenerator, sample_record: dict
) -> None:
    fake.replies["finetuned"] = json.dumps(dict(sample_record["target"], total=1.0))
    res = client.post("/extract", json={"text": sample_record["text"]})
    assert res.status_code == 200
    checks = {c["name"]: c["passed"] for c in res.json()["checks"]}
    assert checks["total_matches_tax"] is False


def test_request_validation(client: TestClient) -> None:
    res = client.post("/extract", json={"text": "short", "model": "gpt"})
    assert res.status_code == 422
    error = res.json()["error"]
    assert error["code"] == "invalid_request"
    assert {tuple(d["loc"]) for d in error["details"]} == {("body", "text"), ("body", "model")}


def test_text_too_long(client: TestClient) -> None:
    res = client.post("/extract", json={"text": "x" * 6000})
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "text_too_long"


def test_unloaded_variant(target_json: str, tmp_path, sample_record: dict) -> None:
    fake = FakeGenerator({"base": target_json}, variants=("base",))
    app = create_app(
        Settings(samples_path=tmp_path / "none.jsonl"), lambda _: InvoiceExtractor(fake)
    )
    with TestClient(app) as c:
        res = c.post("/extract", json={"text": sample_record["text"], "model": "finetuned"})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "model_unavailable"


def test_samples_endpoint(fake: FakeGenerator, tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    record = {"id": "s1", "template": "classic", "text": "abc", "target": {}}
    path.write_text(json.dumps(record) + "\n")
    app = create_app(Settings(samples_path=path), lambda _: InvoiceExtractor(fake))
    with TestClient(app) as c:
        assert c.get("/samples").json() == [{"id": "s1", "template": "classic", "text": "abc"}]
