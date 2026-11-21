import random
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from faker import Faker

from gstlens.data.catalog import (
    NAME_HEADS,
    NAME_SUFFIXES,
    NAME_TRADES,
    PRODUCTS,
    STATE_WEIGHTS,
    STATES,
    STREETS,
    Product,
    State,
)
from gstlens.gstin import make_gstin
from gstlens.schema import Invoice, LineItem

CENT = Decimal("0.01")
DATE_START = date(2023, 4, 1)
DATE_END = date(2025, 8, 31)


def money(value: Decimal | float) -> Decimal:
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


@dataclass
class Party:
    name: str
    gstin: str | None
    address: list[str]
    state: State
    phone: str | None


@dataclass
class Row:
    description: str
    hsn: str
    quantity: Decimal
    unit: str
    rate: Decimal
    gst_rate: int

    @property
    def amount(self) -> Decimal:
        return money(self.quantity * self.rate)


@dataclass
class Draft:
    seller: Party
    buyer: Party
    invoice_number: str
    invoice_date: date
    place_of_supply: State
    rows: list[Row]
    apply_round_off: bool
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def intra_state(self) -> bool:
        return self.seller.state.code == self.place_of_supply.code

    @property
    def taxable_value(self) -> Decimal:
        return sum((r.amount for r in self.rows), Decimal("0"))

    def tax_by_rate(self) -> dict[int, Decimal]:
        out: dict[int, Decimal] = {}
        for r in self.rows:
            out[r.gst_rate] = out.get(r.gst_rate, Decimal("0")) + r.amount
        return dict(sorted(out.items()))

    def half_tax(self, gst_rate: int, base: Decimal) -> Decimal:
        return money(base * gst_rate / Decimal(200))

    def full_tax(self, gst_rate: int, base: Decimal) -> Decimal:
        return money(base * gst_rate / Decimal(100))

    @property
    def cgst(self) -> Decimal:
        if not self.intra_state:
            return Decimal("0.00")
        return sum((self.half_tax(k, v) for k, v in self.tax_by_rate().items()), Decimal("0.00"))

    @property
    def sgst(self) -> Decimal:
        return self.cgst

    @property
    def igst(self) -> Decimal:
        if self.intra_state:
            return Decimal("0.00")
        return sum((self.full_tax(k, v) for k, v in self.tax_by_rate().items()), Decimal("0.00"))

    @property
    def exact_total(self) -> Decimal:
        return self.taxable_value + self.cgst + self.sgst + self.igst

    @property
    def total(self) -> Decimal:
        if self.apply_round_off:
            return self.exact_total.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return self.exact_total

    @property
    def round_off(self) -> Decimal:
        return self.total - self.exact_total

    def with_rows(self, rows: list[Row]) -> "Draft":
        return replace(self, rows=rows)

    def to_invoice(self, **overrides: str) -> Invoice:
        data = {
            "seller_name": self.seller.name,
            "seller_gstin": self.seller.gstin,
            "buyer_name": self.buyer.name,
            "buyer_gstin": self.buyer.gstin,
            "invoice_number": self.invoice_number,
            "invoice_date": self.invoice_date,
            "place_of_supply": self.place_of_supply.name,
            "line_items": [
                LineItem(
                    description=r.description,
                    hsn=r.hsn,
                    quantity=float(r.quantity),
                    rate=float(r.rate),
                    amount=float(r.amount),
                )
                for r in self.rows
            ],
            "taxable_value": float(self.taxable_value),
            "cgst": float(self.cgst),
            "sgst": float(self.sgst),
            "igst": float(self.igst),
            "total": float(self.total),
        }
        data.update(overrides)
        return Invoice.model_validate(data)


class DraftFactory:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.faker = Faker("en_IN")
        self.faker.seed_instance(rng.randrange(2**32))

    def pick_state(self) -> State:
        return self.rng.choices(STATES, weights=STATE_WEIGHTS, k=1)[0]

    def business_name(self) -> str:
        rng = self.rng
        parts = [rng.choice(NAME_HEADS), rng.choice(NAME_TRADES), rng.choice(NAME_SUFFIXES)]
        return " ".join(p for p in parts if p)

    def address(self, state: State) -> list[str]:
        rng = self.rng
        index = rng.randrange(len(state.cities))
        city = state.cities[index]
        pin = state.pin_prefixes[index % len(state.pin_prefixes)] + f"{rng.randint(1, 120):03d}"
        line1 = rng.choice(
            [
                f"{rng.randint(1, 450)}, {rng.choice(STREETS)}",
                f"Shop No. {rng.randint(1, 80)}, {rng.choice(STREETS)}",
                f"Plot {rng.randint(1, 300)}-{rng.choice('ABCD')}, {rng.choice(STREETS)}",
                f"{self.faker.building_number()}, {self.faker.street_name()}",
            ]
        )
        return [line1, f"{city} - {pin}, {state.name}"]

    def phone(self) -> str | None:
        if self.rng.random() < 0.3:
            return None
        return f"+91 {self.rng.randint(70000, 99999)} {self.rng.randint(10000, 99999)}"

    def party(self, state: State, registered: bool, person: bool = False) -> Party:
        name = self.faker.name() if person else self.business_name()
        gstin = make_gstin(state.code, name, self.rng) if registered else None
        return Party(name, gstin, self.address(state), state, self.phone())

    def invoice_number(self, d: date) -> str:
        rng = self.rng
        fy_start = d.year if d.month >= 4 else d.year - 1
        fy = f"{fy_start % 100:02d}-{(fy_start + 1) % 100:02d}"
        n = rng.randint(1, 9999)
        return rng.choice(
            [
                f"INV/{fy}/{n:04d}",
                f"GST-{n}",
                f"TI/{fy}/{n}",
                f"SB{fy_start % 100}{n:05d}",
                f"{fy_start}/{rng.choice(['KA', 'MH', 'TN', 'DL', 'HO'])}/{n:04d}",
                f"{rng.choice(['BLR', 'MUM', 'HYD', 'CHN', 'DEL'])}-INV-{n:05d}",
                f"{n}",
                f"A{rng.randint(1, 9)}-{n:04d}",
            ]
        )

    def row(self, product: Product) -> Row:
        rng = self.rng
        lo, hi = product.price_range
        rate = money(rng.uniform(lo, hi))
        if rate > 1000 and rng.random() < 0.6:
            rate = money(round(float(rate), -1))
        if product.fractional_qty and rng.random() < 0.6:
            qty = Decimal(str(round(rng.uniform(1, 250), rng.choice([1, 2]))))
        elif rate > 20000:
            qty = Decimal(rng.randint(1, 4))
        else:
            qty = Decimal(rng.choice([1, 1, 2, 2, 3, 4, 5, 6, 10, 12, 20, 24, 25, 50, 100]))
        if qty <= 0:
            qty = Decimal(1)
        return Row(product.description, product.hsn, qty, product.unit, rate, product.gst_rate)

    def draft(self) -> Draft:
        rng = self.rng
        seller_state = self.pick_state()
        intra = rng.random() < 0.55
        buyer_state = seller_state if intra else self.pick_state()
        registered = rng.random() < 0.85
        person = not registered and rng.random() < 0.7
        buyer = self.party(buyer_state, registered, person=person)
        seller = self.party(seller_state, True)
        d = DATE_START + timedelta(days=rng.randrange((DATE_END - DATE_START).days))
        n_items = rng.choices([1, 2, 3, 4], weights=[30, 35, 22, 13], k=1)[0]
        if rng.random() < 0.7:
            rate = rng.choice([5, 12, 18, 18, 28])
            pool = [p for p in PRODUCTS if p.gst_rate == rate] or list(PRODUCTS)
        else:
            pool = list(PRODUCTS)
        products = rng.sample(pool, k=min(n_items, len(pool)))
        return Draft(
            seller=seller,
            buyer=buyer,
            invoice_number=self.invoice_number(d),
            invoice_date=d,
            place_of_supply=buyer_state,
            rows=[self.row(p) for p in products],
            apply_round_off=rng.random() < 0.5,
        )


_ONES = (
    "",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Eleven",
    "Twelve",
    "Thirteen",
    "Fourteen",
    "Fifteen",
    "Sixteen",
    "Seventeen",
    "Eighteen",
    "Nineteen",
)
_TENS = ("", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety")


def _below_hundred(n: int) -> str:
    if n < 20:
        return _ONES[n]
    return f"{_TENS[n // 10]} {_ONES[n % 10]}".strip()


def _below_thousand(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    parts = []
    if hundreds:
        parts.append(f"{_ONES[hundreds]} Hundred")
    if rest:
        parts.append(_below_hundred(rest))
    return " ".join(parts)


def number_in_words(n: int) -> str:
    if n == 0:
        return "Zero"
    parts = []
    for size, label in ((10_000_000, "Crore"), (100_000, "Lakh"), (1000, "Thousand")):
        chunk, n = divmod(n, size)
        if chunk:
            parts.append(
                f"{number_in_words(chunk) if chunk >= 100 else _below_hundred(chunk)} {label}"
            )
    if n:
        parts.append(_below_thousand(n))
    return " ".join(parts)


def rupees_in_words(amount: Decimal) -> str:
    rupees = int(amount)
    paise = int((amount - rupees) * 100)
    text = f"Rupees {number_in_words(rupees)}"
    if paise:
        text += f" and {_below_hundred(paise)} Paise"
    return text + " Only"
