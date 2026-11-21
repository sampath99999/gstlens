import argparse
import hashlib
import json
import logging
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

from gstlens.data.generator import DraftFactory
from gstlens.data.templates import HELDOUT_TEMPLATES, TRAIN_TEMPLATES, render

log = logging.getLogger(__name__)

OCR_NOISE_RATE = 0.25


def generate(
    n: int, templates: Sequence[str], seed: int, prefix: str, split: str
) -> Iterator[dict]:
    rng = random.Random(seed)
    factory = DraftFactory(rng)
    for i in range(n):
        template = templates[i % len(templates)]
        text, invoice = render(
            template, factory.draft(), rng, ocr_noise=rng.random() < OCR_NOISE_RATE
        )
        yield {
            "id": f"{prefix}-{i:05d}",
            "split": split,
            "template": template,
            "text": text,
            "target": invoice.to_target_json(),
        }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _dedupe(records: list[dict], seen: set[str]) -> list[dict]:
    out = []
    for r in records:
        key = hashlib.sha1(r["text"].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def build(
    out_dir: Path,
    n_train: int,
    n_val: int,
    n_seen: int,
    n_heldout: int,
    seed: int,
    few_shot_path: Path,
) -> None:
    seen_templates = list(TRAIN_TEMPLATES)
    heldout_templates = list(HELDOUT_TEMPLATES)
    hashes: set[str] = set()
    train = _dedupe(list(generate(n_train, seen_templates, seed, "train", "train")), hashes)
    val = _dedupe(list(generate(n_val, seen_templates, seed + 1, "val", "val")), hashes)
    test = _dedupe(
        list(generate(n_seen, seen_templates, seed + 2, "test-seen", "test_seen"))
        + list(generate(n_heldout, heldout_templates, seed + 3, "test-heldout", "test_heldout")),
        hashes,
    )
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)
    write_jsonl(few_shot_path, select_few_shot(train))
    log.info("wrote train=%d val=%d test=%d to %s", len(train), len(val), len(test), out_dir)


def build_samples(path: Path, seed: int, per_template: int = 2) -> None:
    templates = list(TRAIN_TEMPLATES) + list(HELDOUT_TEMPLATES)
    records = list(
        generate(per_template * len(templates), templates, seed + 100, "sample", "sample")
    )
    write_jsonl(path, records)
    log.info("wrote %d samples to %s", len(records), path)


def select_few_shot(train: list[dict]) -> list[dict]:
    """Two short examples: an intra-state classic invoice and an inter-state key-value one."""

    def pick(template: str, intra: bool) -> dict:
        candidates = [
            r
            for r in train
            if r["template"] == template
            and (r["target"]["cgst"] > 0) == intra
            and r["target"]["buyer_gstin"]
            and len(r["target"]["line_items"]) == 2
        ]
        return min(candidates, key=lambda r: len(r["text"]))

    return [pick("classic", True), pick("key_value", False)]


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Generate synthetic GST invoice datasets")
    parser.add_argument("--out", type=Path, default=Path("data/generated"))
    parser.add_argument("--train", type=int, default=2400)
    parser.add_argument("--val", type=int, default=150)
    parser.add_argument("--test-seen", type=int, default=100)
    parser.add_argument("--test-heldout", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=Path, default=Path("data/samples.jsonl"))
    parser.add_argument("--few-shot", type=Path, default=Path("data/few_shot.jsonl"))
    args = parser.parse_args(argv)
    build(
        args.out, args.train, args.val, args.test_seen, args.test_heldout, args.seed, args.few_shot
    )
    build_samples(args.samples, args.seed)


if __name__ == "__main__":
    main()
