from collections.abc import Sequence
from pathlib import Path

from gstlens.evaluation.metrics import Summary

LABELS = {
    "base-zero": "Qwen2.5-0.5B, zero-shot",
    "base-few": "Qwen2.5-0.5B, 2-shot",
    "finetuned": "Qwen2.5-0.5B + LoRA",
    "ollama-few": "qwen2.5:3b (Ollama Q4), 2-shot",
}
SPLIT_TITLES = {
    "test_heldout": "Held-out layout (e-invoice print, never seen in training)",
    "test_seen": "Seen layouts (same six templates as training, new invoices)",
}


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _overview(results: dict[str, dict[str, Summary]], split: str) -> list[str]:
    n = next(iter(r[split].n for r in results.values() if split in r))
    lines = [
        f"### {SPLIT_TITLES[split]} (n={n})",
        "",
        "| System | Raw valid JSON | JSON after repair | Schema valid | Field F1 "
        "| Line-item F1 | All items exact | Whole invoice exact | Latency p50 | Latency p95 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for system, splits in results.items():
        s = splits.get(split)
        if s is None:
            continue
        lines.append(
            f"| {LABELS.get(system, system)} | {pct(s.raw_valid_json)} "
            f"| {pct(s.valid_json_after_repair)} | {pct(s.schema_valid)} | {s.field_f1:.3f} "
            f"| {s.item_f1:.3f} | {pct(s.items_exact)} | {pct(s.doc_exact)} "
            f"| {s.latency_p50_s:.1f}s | {s.latency_p95_s:.1f}s |"
        )
    return [*lines, ""]


def _per_field(results: dict[str, dict[str, Summary]], fields: Sequence[str]) -> list[str]:
    systems = list(results)
    lines = [
        "### Exact match per field (held-out / seen)",
        "",
        "| Field | " + " | ".join(LABELS.get(s, s) for s in systems) + " |",
        "|---|" + "---|" * len(systems),
    ]
    for name in fields:
        cells = []
        for system in systems:
            held = results[system].get("test_heldout")
            seen = results[system].get("test_seen")
            h = pct(held.field_accuracy[name]) if held else "-"
            s = pct(seen.field_accuracy[name]) if seen else "-"
            cells.append(f"{h} / {s}")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return [*lines, ""]


def _per_template(by_template: dict[str, dict[str, Summary]]) -> list[str]:
    systems = list(by_template)
    templates = sorted({t for groups in by_template.values() for t in groups})
    lines = [
        "### Whole invoice exact match by layout",
        "",
        "| Layout | n | " + " | ".join(LABELS.get(s, s) for s in systems) + " |",
        "|---|---|" + "---|" * len(systems),
    ]
    for template in templates:
        n = next(g[template].n for g in by_template.values() if template in g)
        cells = [
            pct(by_template[s][template].doc_exact) if template in by_template[s] else "-"
            for s in systems
        ]
        lines.append(f"| {template} | {n} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def write_report(
    results: dict[str, dict[str, Summary]],
    by_template: dict[str, dict[str, Summary]],
    path: Path,
    fields: Sequence[str],
) -> None:
    lines: list[str] = []
    for split in ("test_heldout", "test_seen"):
        if any(split in r for r in results.values()):
            lines += _overview(results, split)
    lines += _per_field(results, fields)
    lines += _per_template(by_template)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
