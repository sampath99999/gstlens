import json
import random

import pytest

from gstlens.data.generator import DraftFactory
from gstlens.data.templates import render
from gstlens.extraction.base import Generation, Variant


class FakeGenerator:
    def __init__(
        self, replies: dict[Variant, str], variants: tuple[Variant, ...] = ("base", "finetuned")
    ) -> None:
        self.replies = replies
        self._variants = variants
        self.calls: list[tuple[list[dict[str, str]], Variant]] = []

    @property
    def variants(self) -> tuple[Variant, ...]:
        return self._variants

    def generate(self, messages: list[dict[str, str]], variant: Variant) -> Generation:
        self.calls.append((messages, variant))
        return Generation(
            self.replies[variant], latency_s=0.25, prompt_tokens=100, completion_tokens=50
        )


@pytest.fixture
def sample_record() -> dict:
    rng = random.Random(3)
    text, invoice = render("classic", DraftFactory(rng).draft(), rng, ocr_noise=False)
    return {"text": text, "target": invoice.to_target_json()}


@pytest.fixture
def target_json(sample_record: dict) -> str:
    return json.dumps(sample_record["target"])
