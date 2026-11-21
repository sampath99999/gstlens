from dataclasses import dataclass
from typing import Literal, Protocol

Variant = Literal["base", "finetuned"]


@dataclass(frozen=True)
class Generation:
    text: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int


class TextGenerator(Protocol):
    @property
    def variants(self) -> tuple[Variant, ...]: ...

    def generate(self, messages: list[dict[str, str]], variant: Variant) -> Generation: ...
