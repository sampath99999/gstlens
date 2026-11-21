import random
from decimal import Decimal

import pytest

from gstlens.data.build import generate, select_few_shot
from gstlens.data.generator import DraftFactory, number_in_words, rupees_in_words
from gstlens.data.templates import ALL_TEMPLATES, HELDOUT_TEMPLATES, TRAIN_TEMPLATES, render
from gstlens.gstin import is_valid_gstin
from gstlens.schema import Invoice


@pytest.fixture(scope="module")
def drafts():
    factory = DraftFactory(random.Random(11))
    return [factory.draft() for _ in range(300)]


def test_totals_add_up(drafts) -> None:
    for d in drafts:
        assert d.taxable_value == sum(r.amount for r in d.rows)
        assert d.taxable_value + d.cgst + d.sgst + d.igst + d.round_off == d.total
        assert abs(d.round_off) <= Decimal("0.5")


def test_tax_regime_follows_place_of_supply(drafts) -> None:
    for d in drafts:
        if d.intra_state:
            assert d.igst == 0
            assert d.cgst == d.sgst > 0
        else:
            assert d.cgst == d.sgst == 0
            assert d.igst > 0


def test_parties_have_valid_gstins(drafts) -> None:
    for d in drafts:
        assert is_valid_gstin(d.seller.gstin)
        assert d.seller.gstin.startswith(d.seller.state.code)
        if d.buyer.gstin:
            assert is_valid_gstin(d.buyer.gstin)
            assert d.buyer.gstin.startswith(d.buyer.state.code)


@pytest.mark.parametrize("template", sorted(ALL_TEMPLATES))
@pytest.mark.parametrize("noise", [False, True])
def test_every_template_renders_a_consistent_target(template: str, noise: bool) -> None:
    rng = random.Random(5)
    factory = DraftFactory(rng)
    for _ in range(20):
        text, invoice = render(template, factory.draft(), rng, ocr_noise=noise)
        assert "\x01" not in text and "\x02" not in text
        assert invoice.seller_gstin in text
        assert invoice.invoice_number in text
        items = round(sum(i.amount for i in invoice.line_items), 2)
        assert items == pytest.approx(invoice.taxable_value)
        tax = invoice.taxable_value + invoice.cgst + invoice.sgst + invoice.igst
        assert abs(tax - invoice.total) <= 0.5
        for item in invoice.line_items:
            assert item.hsn in text
            assert item.amount == pytest.approx(round(item.quantity * item.rate, 2))
        Invoice.model_validate(invoice.to_target_json())


def test_generation_is_deterministic() -> None:
    a = list(generate(10, list(TRAIN_TEMPLATES), 42, "x", "train"))
    b = list(generate(10, list(TRAIN_TEMPLATES), 42, "x", "train"))
    assert a == b


def test_heldout_template_is_not_in_training_set() -> None:
    assert set(HELDOUT_TEMPLATES).isdisjoint(TRAIN_TEMPLATES)
    records = list(generate(30, list(TRAIN_TEMPLATES), 1, "t", "train"))
    assert {r["template"] for r in records} == set(TRAIN_TEMPLATES)


def test_few_shot_selection_covers_both_tax_regimes() -> None:
    train = list(generate(120, list(TRAIN_TEMPLATES), 2, "t", "train"))
    first, second = select_few_shot(train)
    assert first["target"]["cgst"] > 0
    assert second["target"]["igst"] > 0


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "Zero"),
        (15, "Fifteen"),
        (1062, "One Thousand Sixty Two"),
        (250000, "Two Lakh Fifty Thousand"),
        (12345678, "One Crore Twenty Three Lakh Forty Five Thousand Six Hundred Seventy Eight"),
    ],
)
def test_number_in_words_uses_indian_system(n: int, words: str) -> None:
    assert number_in_words(n) == words


def test_rupees_in_words_includes_paise() -> None:
    assert rupees_in_words(Decimal("101.50")) == "Rupees One Hundred One and Fifty Paise Only"
