from time import perf_counter

from openai import OpenAI

from gstlens.extraction.base import Generation, Variant


class OpenAICompatGenerator:
    """Any OpenAI-compatible chat endpoint (Ollama, vLLM, llama.cpp server)."""

    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int = 640) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    @property
    def variants(self) -> tuple[Variant, ...]:
        return ("base",)

    def generate(self, messages: list[dict[str, str]], variant: Variant) -> Generation:
        start = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0,
            max_tokens=self.max_tokens,
        )
        latency = perf_counter() - start
        usage = response.usage
        return Generation(
            text=response.choices[0].message.content or "",
            latency_s=latency,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
