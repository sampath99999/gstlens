import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from gstlens.data.build import read_jsonl
from gstlens.prompts import build_messages, target_to_text

log = logging.getLogger(__name__)

IGNORE_INDEX = -100
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    train_file: str = "data/generated/train.jsonl"
    val_file: str = "data/generated/val.jsonl"
    output_dir: str = "artifacts/qwen05b-lora"
    max_length: int = 1024
    epochs: float = 1.0
    max_steps: int = -1
    limit: int | None = None
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_accum: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    warmup_ratio: float = 0.03
    eval_examples: int = 48
    eval_steps: int = 100
    save_steps: int = 25
    seed: int = 7
    fp16: bool = True
    resume: bool = False


def encode_example(
    tokenizer: PreTrainedTokenizerBase, record: dict, max_length: int
) -> dict[str, list[int]] | None:
    prompt = tokenizer.apply_chat_template(
        build_messages(record["text"]), add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(
        target_to_text(record["target"]) + "<|im_end|>", add_special_tokens=False
    )["input_ids"]
    input_ids = prompt_ids + completion_ids
    if len(input_ids) > max_length:
        return None
    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
    return {"input_ids": input_ids, "labels": labels}


class InvoiceSFTDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer: PreTrainedTokenizerBase, max_length: int):
        encoded = [encode_example(tokenizer, r, max_length) for r in records]
        self.items = [e for e in encoded if e is not None]
        self.dropped = len(encoded) - len(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.items[idx]


class PadCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_token_id] * pad)
            labels.append(b["labels"] + [IGNORE_INDEX] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


class CompletionOnlyTrainer(Trainer):
    """Projects to the vocabulary only at positions that carry a label.

    Qwen's 152k-token vocabulary makes full-sequence logits the largest tensor in the step
    (about 0.6 GB in fp32 at 1024 tokens). Prompt positions have no loss, so skipping them
    is what lets a 1024-token step fit on a 4 GB card.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.pop("labels")
        shifted = labels[:, 1:]
        keep = (shifted != IGNORE_INDEX).any(dim=0).nonzero().squeeze(-1)
        outputs = model(**inputs, logits_to_keep=keep)
        logits = outputs.logits.float()
        targets = shifted[:, keep]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=IGNORE_INDEX
        )
        return (loss, outputs) if return_outputs else loss


class JsonlLogger(TrainerCallback):
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **_: Any):
        if logs:
            with self.path.open("a") as f:
                f.write(json.dumps({"step": state.global_step, **logs}) + "\n")


def train(cfg: TrainConfig) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    train_records = read_jsonl(Path(cfg.train_file))
    if cfg.limit:
        train_records = train_records[: cfg.limit]
    val_records = read_jsonl(Path(cfg.val_file))[: cfg.eval_examples]
    train_ds = InvoiceSFTDataset(train_records, tokenizer, cfg.max_length)
    val_ds = InvoiceSFTDataset(val_records, tokenizer, cfg.max_length)
    log.info(
        "train examples=%d (dropped %d over %d tokens), val examples=%d",
        len(train_ds),
        train_ds.dropped,
        cfg.max_length,
        len(val_ds),
    )

    use_fp16 = cfg.fp16 and torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        dtype=torch.float16 if use_fp16 else torch.float32,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    # Trainable LoRA weights stay in fp32 so the fp16 GradScaler can unscale them.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    trainable, total = model.get_nb_trainable_parameters()
    log.info("trainable params %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=cfg.warmup_ratio,
        weight_decay=0.0,
        max_grad_norm=1.0,
        fp16=use_fp16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        report_to=[],
        seed=cfg.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = CompletionOnlyTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[JsonlLogger(out_dir / "train_log.jsonl")],
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ckpt_dir = Path(args.output_dir)
    checkpoint = get_last_checkpoint(str(ckpt_dir)) if cfg.resume and ckpt_dir.is_dir() else None
    if checkpoint:
        log.info("resuming from %s", checkpoint)
    start = time.time()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    elapsed = time.time() - start
    metrics = trainer.evaluate()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    summary = {
        "config": asdict(cfg),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "train_examples": len(train_ds),
        "dropped_too_long": train_ds.dropped,
        "steps": result.global_step,
        "resumed_from": checkpoint,
        "train_runtime_s": round(elapsed, 1),
        "train_loss": round(result.training_loss, 4),
        "eval_loss": round(metrics["eval_loss"], 4),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available()
        else None,
        "trainable_params": trainable,
        "total_params": total,
        "torch": torch.__version__,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("saved adapter to %s: %s", adapter_dir, summary)
    return summary


def parse_args(argv: Sequence[str] | None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="LoRA fine-tune for GST invoice extraction")
    defaults = TrainConfig()
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
        else:
            kind = type(value) if value is not None else int
            parser.add_argument(flag, type=kind, default=value)
    return TrainConfig(**vars(parser.parse_args(argv)))


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    train(parse_args(argv))


if __name__ == "__main__":
    main()
