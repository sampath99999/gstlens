from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GSTLENS_", env_file=".env", extra="ignore")

    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path: Path | None = Path("artifacts/qwen05b-lora/adapter")
    device: str = "auto"
    dtype: Literal["float16", "float32"] = "float16"
    max_new_tokens: int = 640
    backend: Literal["hf", "fake"] = "hf"
    samples_path: Path = Path("data/samples.jsonl")
    max_input_chars: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = "ollama"
    openai_model: str = "qwen2.5:3b"


@lru_cache
def get_settings() -> Settings:
    return Settings()
