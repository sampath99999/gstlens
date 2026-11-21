import re
import statistics
from dataclasses import dataclass, field
from datetime import date
from typing import Any

STRING_FIELDS = ("seller_name", "buyer_name", "invoice_number", "place_of_supply")
GSTIN_FIELDS = ("seller_gstin", "buyer_gstin")
DATE_FIELDS = ("invoice_date",)
NUMBER_FIELDS = ("taxable_value", "cgst", "sgst", "igst", "total")
SCALAR_FIELDS = (
    "seller_name",
    "seller_gstin",
    "buyer_name",
    "buyer_gstin",
    "invoice_number",
    "invoice_date",
    "place_of_supply",
    "taxable_value",
    "cgst",
    "sgst",
    "igst",
    "total",
)
ITEM_NUMBER_FIELDS = ("quantity", "rate", "amount")
MONEY_TOLERANCE = 0.011

_NON_NUMERIC_RE = re.compile(r"[^\d.\-]")
_STATE_CODE_RE = re.compile(r"[\d()\-]")


def to_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        cleaned = _NON_NUMERIC_RE.sub("", value.replace("Rs.", ""))
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).casefold()
    return text or None


def normalize(field_name: str, value: Any) -> Any:
    if field_name in NUMBER_FIELDS or field_name in ITEM_NUMBER_FIELDS:
        return to_number(value)
    text = _text(value)
    if text is None or text in {"null", "none", "n/a", "-"}:
        return None
    if field_name in GSTIN_FIELDS:
        return text.replace(" ", "").upper()
    if field_name in DATE_FIELDS:
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            return f"invalid:{text}"
    if field_name == "place_of_supply":
        return _STATE_CODE_RE.sub("", text).strip()
    if field_name in ("invoice_number", "hsn"):
        return text.replace(" ", "")
    return text


def values_match(field_name: str, gold: Any, pred: Any) -> bool:
    g, p = normalize(field_name, gold), normalize(field_name, pred)
    if isinstance(g, float) and isinstance(p, float):
        return abs(g - p) <= MONEY_TOLERANCE
    return g == p


def item_matches(gold: dict, pred: Any) -> bool:
    if not isinstance(pred, dict):
        return False
    return all(
        values_match(name, gold.get(name), pred.get(name))
        for name in ("description", "hsn", *ITEM_NUMBER_FIELDS)
    )


@dataclass
class DocScore:
    field_correct: dict[str, bool]
    tp: int
    fp: int
    fn: int
    items_gold: int
    items_pred: int
    items_matched: int

    @property
    def items_exact(self) -> bool:
        return self.items_matched == self.items_gold == self.items_pred

    @property
    def doc_exact(self) -> bool:
        return self.items_exact and all(self.field_correct.values())


def score_document(gold: dict, pred: dict | None) -> DocScore:
    pred = pred or {}
    correct: dict[str, bool] = {}
    tp = fp = fn = 0
    for name in SCALAR_FIELDS:
        g = normalize(name, gold.get(name))
        p = normalize(name, pred.get(name))
        ok = values_match(name, gold.get(name), pred.get(name))
        correct[name] = ok
        if p is not None and ok:
            tp += 1
        elif p is not None:
            fp += 1
            fn += g is not None
        elif g is not None:
            fn += 1

    gold_items = gold.get("line_items") or []
    pred_items = pred.get("line_items")
    pred_items = pred_items if isinstance(pred_items, list) else []
    used: set[int] = set()
    matched = 0
    for g_item in gold_items:
        for j, p_item in enumerate(pred_items):
            if j not in used and item_matches(g_item, p_item):
                used.add(j)
                matched += 1
                break
    return DocScore(correct, tp, fp, fn, len(gold_items), len(pred_items), matched)


def f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


@dataclass
class Prediction:
    gold: dict
    parsed: dict | None
    raw_valid: bool
    schema_valid: bool
    latency_s: float
    completion_tokens: int


@dataclass
class Summary:
    n: int
    raw_valid_json: float
    valid_json_after_repair: float
    schema_valid: float
    field_f1: float
    field_accuracy: dict[str, float]
    item_f1: float
    items_exact: float
    doc_exact: float
    latency_mean_s: float
    latency_p50_s: float
    latency_p95_s: float
    completion_tokens_mean: float
    extra: dict[str, Any] = field(default_factory=dict)


def summarize(predictions: list[Prediction]) -> Summary:
    n = len(predictions)
    if n == 0:
        raise ValueError("no predictions to summarize")
    scores = [score_document(p.gold, p.parsed) for p in predictions]
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    matched = sum(s.items_matched for s in scores)
    items_pred = sum(s.items_pred for s in scores)
    items_gold = sum(s.items_gold for s in scores)
    latencies = [p.latency_s for p in predictions]
    return Summary(
        n=n,
        raw_valid_json=sum(p.raw_valid for p in predictions) / n,
        valid_json_after_repair=sum(p.parsed is not None for p in predictions) / n,
        schema_valid=sum(p.schema_valid for p in predictions) / n,
        field_f1=f1(tp, fp, fn),
        field_accuracy={
            name: sum(s.field_correct[name] for s in scores) / n for name in SCALAR_FIELDS
        },
        item_f1=f1(matched, items_pred - matched, items_gold - matched),
        items_exact=sum(s.items_exact for s in scores) / n,
        doc_exact=sum(s.doc_exact for s in scores) / n,
        latency_mean_s=statistics.fmean(latencies),
        latency_p50_s=percentile(latencies, 0.5),
        latency_p95_s=percentile(latencies, 0.95),
        completion_tokens_mean=statistics.fmean(p.completion_tokens for p in predictions),
    )
