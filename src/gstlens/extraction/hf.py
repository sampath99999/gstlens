import logging
import threading
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from gstlens.extraction.base import Generation, Variant

log = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class HFGenerator:
    """Local Hugging Face model, optionally with a LoRA adapter.

    With merge=False the adapter stays separate so one copy of the weights serves both
    variants (the adapter is disabled for "base"). With merge=True the adapter is folded
    into the weights for faster decoding and only the "finetuned" variant is served.
    """

    def __init__(
        self,
        base_model: str,
        adapter_path: Path | None = None,
        device: str = "auto",
        dtype: str = "float16",
        max_new_tokens: int = 640,
        merge: bool = False,
    ) -> None:
        self.device = resolve_device(device)
        torch_dtype = getattr(torch, dtype) if self.device != "cpu" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch_dtype)
        self._has_adapter = False
        if adapter_path is not None:
            if not Path(adapter_path).exists():
                raise FileNotFoundError(f"LoRA adapter not found at {adapter_path}")
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
            if merge:
                model = model.merge_and_unload()
            self._has_adapter = True
        self._merged = merge and self._has_adapter
        self.model = model.to(self.device).eval()
        eos = [self.tokenizer.convert_tokens_to_ids("<|im_end|>"), self.tokenizer.eos_token_id]
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        self._lock = threading.Lock()
        log.info(
            "loaded %s on %s (%s), adapter=%s merged=%s",
            base_model,
            self.device,
            torch_dtype,
            adapter_path,
            self._merged,
        )

    @property
    def variants(self) -> tuple[Variant, ...]:
        if self._merged:
            return ("finetuned",)
        return ("base", "finetuned") if self._has_adapter else ("base",)

    def generate(self, messages: list[dict[str, str]], variant: Variant) -> Generation:
        if variant not in self.variants:
            raise ValueError(f"variant '{variant}' is not loaded")
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        use_base = variant == "base" and self._has_adapter and not self._merged
        adapter_ctx = self.model.disable_adapter() if use_base else nullcontext()
        with self._lock, torch.inference_mode(), adapter_ctx:
            start = perf_counter()
            output = self.model.generate(**inputs, generation_config=self.generation_config)
            if self.device == "cuda":
                torch.cuda.synchronize()
            latency = perf_counter() - start
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return Generation(text, latency, prompt_len, int(new_tokens.shape[0]))
