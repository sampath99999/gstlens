import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from pydantic import ValidationError

from gstlens.config import get_settings
from gstlens.data.build import read_jsonl
from gstlens.evaluation.metrics import SCALAR_FIELDS, Prediction, Summary, summarize
from gstlens.evaluation.report import write_report
from gstlens.extraction.base import TextGenerator, Variant
from gstlens.extraction.parsing import parse_model_output
from gstlens.prompts import build_messages
from gstlens.schema import Invoice

log = logging.getLogger(__name__)

SYSTEMS = ("base-zero", "base-few", "finetuned", "ollama-few")
SPLITS = ("test_heldout", "test_seen")


def load_generator(system: str, adapter: Path) -> tuple[TextGenerator, Variant, bool]:
    settings = get_settings()
    if system.startswith("ollama"):
        from gstlens.extraction.openai_compat import OpenAICompatGenerator

        gen = OpenAICompatGenerator(
            settings.openai_base_url, settings.openai_api_key, settings.openai_model
        )
        return gen, "base", system.endswith("few")

    from gstlens.extraction.hf import HFGenerator

    if system == "finetuned":
        gen = HFGenerator(settings.base_model, adapter, settings.device, settings.dtype, merge=True)
        return gen, "finetuned", False
    gen = HFGenerator(settings.base_model, None, settings.device, settings.dtype)
    return gen, "base", system == "base-few"


def run_system(
    system: str,
    test_file: Path,
    few_shot_file: Path,
    out_dir: Path,
    adapter: Path,
    limit: int | None,
) -> Path:
    records = read_jsonl(test_file)
    if limit:
        per_split = {s: [r for r in records if r["split"] == s][:limit] for s in SPLITS}
        records = [r for s in SPLITS for r in per_split[s]]
    out_path = out_dir / f"{system}.jsonl"
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in records if r["id"] not in done]
    if not todo:
        log.info("%s: all %d predictions already present", system, len(records))
        return out_path

    generator, variant, use_few_shot = load_generator(system, adapter)
    few_shot = read_jsonl(few_shot_file) if use_few_shot else []
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for i, record in enumerate(todo, 1):
            gen = generator.generate(build_messages(record["text"], few_shot), variant)
            row = {
                "id": record["id"],
                "split": record["split"],
                "template": record["template"],
                "raw_output": gen.text,
                "latency_s": round(gen.latency_s, 4),
                "prompt_tokens": gen.prompt_tokens,
                "completion_tokens": gen.completion_tokens,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                log.info("%s: %d/%d (last %.1fs)", system, i, len(todo), gen.latency_s)
    return out_path


def to_prediction(gold: dict, row: dict) -> Prediction:
    parsed = parse_model_output(row["raw_output"])
    schema_valid = False
    if parsed.data is not None:
        try:
            Invoice.model_validate(parsed.data)
            schema_valid = True
        except ValidationError:
            pass
    return Prediction(
        gold=gold,
        parsed=parsed.data,
        raw_valid=parsed.raw_valid,
        schema_valid=schema_valid,
        latency_s=row["latency_s"],
        completion_tokens=row["completion_tokens"],
    )


def score_all(test_file: Path, pred_dir: Path) -> dict[str, dict[str, Summary]]:
    gold = {r["id"]: r for r in read_jsonl(test_file)}
    results: dict[str, dict[str, Summary]] = {}
    for system in SYSTEMS:
        path = pred_dir / f"{system}.jsonl"
        if not path.exists():
            continue
        rows = [r for r in read_jsonl(path) if r["id"] in gold]
        results[system] = {}
        for split in (*SPLITS, "all"):
            subset = [r for r in rows if split == "all" or r["split"] == split]
            if subset:
                preds = [to_prediction(gold[r["id"]]["target"], r) for r in subset]
                results[system][split] = summarize(preds)
    return results


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Evaluate extraction systems on the test split")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS[:3]))
    run.add_argument("--limit", type=int, default=None, help="max examples per split")
    run.add_argument("--adapter", type=Path, default=Path("artifacts/qwen05b-lora/adapter"))
    report = sub.add_parser("report")
    for p in (run, report):
        p.add_argument("--test-file", type=Path, default=Path("data/generated/test.jsonl"))
        p.add_argument("--pred-dir", type=Path, default=Path("artifacts/eval"))
    run.add_argument("--few-shot-file", type=Path, default=Path("data/few_shot.jsonl"))
    report.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    if args.command == "run":
        for system in args.systems:
            run_system(
                system, args.test_file, args.few_shot_file, args.pred_dir, args.adapter, args.limit
            )
        return

    results = score_all(args.test_file, args.pred_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {s: {k: asdict(v) for k, v in splits.items()} for s, splits in results.items()}
    (args.out / "metrics.json").write_text(json.dumps(payload, indent=2))
    write_report(results, args.out / "results.md", SCALAR_FIELDS)
    log.info("wrote %s", args.out / "results.md")


if __name__ == "__main__":
    main()
