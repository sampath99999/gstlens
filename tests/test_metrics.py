import copy

import pytest

from gstlens.evaluation.metrics import (
    Prediction,
    f1,
    normalize,
    percentile,
    score_document,
    summarize,
    values_match,
)


@pytest.mark.parametrize(
    ("field", "gold", "pred", "expected"),
    [
        ("total", 1234.5, "Rs. 1,234.50", True),
        ("total", 1234.5, 1234.51, True),
        ("total", 1234.5, 1234.6, False),
        ("seller_name", "Sri Balaji Traders", "  sri  balaji TRADERS ", True),
        ("place_of_supply", "Karnataka", "29-Karnataka", True),
        ("invoice_date", "2024-03-12", "12/03/2024", False),
        ("invoice_date", "2024-03-12", "2024-03-12", True),
        ("buyer_gstin", None, "null", True),
        ("buyer_gstin", None, "29AAGCB7383J1Z4", False),
        ("seller_gstin", "29AAGCB7383J1Z4", "29aagcb7383j1z4", True),
    ],
)
def test_values_match(field: str, gold: object, pred: object, expected: bool) -> None:
    assert values_match(field, gold, pred) is expected


def test_normalize_number_rejects_text() -> None:
    assert normalize("cgst", "nil") is None


def test_perfect_prediction(sample_record: dict) -> None:
    gold = sample_record["target"]
    score = score_document(gold, copy.deepcopy(gold))
    assert score.doc_exact and score.items_exact
    assert score.fp == score.fn == 0


def test_wrong_value_counts_as_false_positive_and_negative(sample_record: dict) -> None:
    gold = sample_record["target"]
    pred = {**copy.deepcopy(gold), "total": gold["total"] + 10}
    score = score_document(gold, pred)
    assert not score.field_correct["total"]
    assert (score.fp, score.fn) == (1, 1)
    assert not score.doc_exact


def test_missing_prediction_is_all_false_negatives(sample_record: dict) -> None:
    gold = sample_record["target"]
    score = score_document(gold, None)
    present = sum(v is not None for k, v in gold.items() if k != "line_items")
    assert score.tp == 0 and score.fn == present
    assert score.items_matched == 0


def test_line_items_match_in_any_order(sample_record: dict) -> None:
    gold = copy.deepcopy(sample_record["target"])
    gold["line_items"].append(
        {"description": "Extra", "hsn": "1234", "quantity": 1, "rate": 5, "amount": 5}
    )
    pred = copy.deepcopy(gold)
    pred["line_items"].reverse()
    assert score_document(gold, pred).items_exact


def test_extra_predicted_item_breaks_exact_match(sample_record: dict) -> None:
    gold = sample_record["target"]
    pred = copy.deepcopy(gold)
    pred["line_items"].append(dict(pred["line_items"][0]))
    score = score_document(gold, pred)
    assert score.items_matched == len(gold["line_items"])
    assert not score.items_exact


def test_f1_and_percentile() -> None:
    assert f1(8, 2, 2) == pytest.approx(0.8)
    assert f1(0, 0, 0) == 0.0
    assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)


def test_summarize(sample_record: dict) -> None:
    gold = sample_record["target"]
    preds = [
        Prediction(gold, copy.deepcopy(gold), True, True, 1.0, 100),
        Prediction(gold, None, False, False, 3.0, 300),
    ]
    s = summarize(preds)
    assert s.raw_valid_json == 0.5
    assert s.valid_json_after_repair == 0.5
    assert s.doc_exact == 0.5
    assert s.latency_mean_s == 2.0
    assert 0 < s.field_f1 < 1
