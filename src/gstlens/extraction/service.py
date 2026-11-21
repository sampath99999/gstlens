from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from gstlens.extraction.base import Generation, TextGenerator, Variant
from gstlens.extraction.parsing import ParsedOutput, parse_model_output
from gstlens.gstin import is_valid_gstin
from gstlens.prompts import build_messages
from gstlens.schema import Invoice


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Extraction:
    variant: Variant
    generation: Generation
    parsed: ParsedOutput
    invoice: Invoice | None
    errors: list[dict[str, Any]] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)


def consistency_checks(invoice: Invoice) -> list[Check]:
    checks = [
        Check("seller_gstin_checksum", is_valid_gstin(invoice.seller_gstin), invoice.seller_gstin)
    ]
    if invoice.buyer_gstin:
        checks.append(
            Check("buyer_gstin_checksum", is_valid_gstin(invoice.buyer_gstin), invoice.buyer_gstin)
        )
    items_sum = round(sum(i.amount for i in invoice.line_items), 2)
    checks.append(
        Check(
            "items_sum_to_taxable",
            abs(items_sum - invoice.taxable_value) < 0.02,
            f"items {items_sum:.2f} vs taxable {invoice.taxable_value:.2f}",
        )
    )
    computed = invoice.taxable_value + invoice.cgst + invoice.sgst + invoice.igst
    checks.append(
        Check(
            "total_matches_tax",
            abs(computed - invoice.total) <= 1.0,
            f"taxable + tax {computed:.2f} vs total {invoice.total:.2f}",
        )
    )
    one_regime = invoice.igst == 0 or (invoice.cgst == 0 and invoice.sgst == 0)
    checks.append(Check("single_tax_regime", one_regime, "either IGST or CGST+SGST"))
    return checks


class InvoiceExtractor:
    def __init__(
        self, generator: TextGenerator, few_shot: Mapping[Variant, list[dict]] | None = None
    ) -> None:
        self.generator = generator
        self.few_shot = dict(few_shot or {})

    @property
    def variants(self) -> tuple[Variant, ...]:
        return self.generator.variants

    def extract(self, text: str, variant: Variant) -> Extraction:
        messages = build_messages(text, self.few_shot.get(variant))
        generation = self.generator.generate(messages, variant)
        parsed = parse_model_output(generation.text)
        if parsed.data is None:
            error = {"loc": [], "msg": parsed.error or "invalid JSON", "type": "json_invalid"}
            return Extraction(variant, generation, parsed, None, errors=[error])
        try:
            invoice = Invoice.model_validate(parsed.data)
        except ValidationError as exc:
            errors = [
                {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
            ]
            return Extraction(variant, generation, parsed, None, errors=errors)
        return Extraction(variant, generation, parsed, invoice, checks=consistency_checks(invoice))
