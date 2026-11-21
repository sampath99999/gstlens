import pytest
from pydantic import ValidationError

from gstlens.schema import Invoice


def test_target_round_trips(sample_record: dict) -> None:
    invoice = Invoice.model_validate(sample_record["target"])
    assert invoice.to_target_json() == sample_record["target"]


def test_gstin_is_normalised_to_upper_case(sample_record: dict) -> None:
    target = sample_record["target"]
    data = {**target, "seller_gstin": target["seller_gstin"].lower()}
    assert Invoice.model_validate(data).seller_gstin == target["seller_gstin"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seller_gstin", "27AAPFU0939F1"),
        ("invoice_date", "31/02/2024"),
        ("total", -5),
        ("line_items", []),
    ],
)
def test_rejects_invalid_fields(sample_record: dict, field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Invoice.model_validate({**sample_record["target"], field: value})


def test_rejects_unknown_keys(sample_record: dict) -> None:
    with pytest.raises(ValidationError):
        Invoice.model_validate({**sample_record["target"], "notes": "x"})


def test_buyer_gstin_may_be_null(sample_record: dict) -> None:
    invoice = Invoice.model_validate({**sample_record["target"], "buyer_gstin": None})
    assert invoice.buyer_gstin is None


def test_hsn_must_be_digits(sample_record: dict) -> None:
    items = [dict(sample_record["target"]["line_items"][0], hsn="1.5 Ton")]
    with pytest.raises(ValidationError):
        Invoice.model_validate({**sample_record["target"], "line_items": items})
