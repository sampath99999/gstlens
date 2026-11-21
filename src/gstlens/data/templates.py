import random
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal

from gstlens.data.generator import Draft, Party, Row, rupees_in_words
from gstlens.gstin import make_gstin
from gstlens.schema import Invoice

LABEL_OPEN, LABEL_CLOSE = "\x01", "\x02"
LABEL_RE = re.compile(f"{LABEL_OPEN}(.*?){LABEL_CLOSE}")

DATE_FORMATS = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d-%b-%Y",
    "%d %B %Y",
    "%d.%m.%Y",
    "%Y-%m-%d",
    "%d/%m/%y",
    "%b %d, %Y",
    "%d-%b-%y",
)


def L(label: str) -> str:
    return f"{LABEL_OPEN}{label}{LABEL_CLOSE}"


def indian_grouping(value: Decimal) -> str:
    sign = "-" if value < 0 else ""
    whole, frac = f"{abs(value):.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join([*groups, tail])
    return f"{sign}{whole}.{frac}"


@dataclass(frozen=True)
class Style:
    grouping: str
    currency: str
    date_format: str

    @classmethod
    def random(cls, rng: random.Random) -> "Style":
        return cls(
            grouping=rng.choice(["indian", "indian", "western", "plain"]),
            currency=rng.choice(["", "", "Rs. ", "Rs.", "₹", "₹ ", "INR "]),
            date_format=rng.choice(DATE_FORMATS),
        )

    def num(self, value: Decimal) -> str:
        if self.grouping == "indian":
            return indian_grouping(value)
        if self.grouping == "western":
            return f"{value:,.2f}"
        return f"{value:.2f}"

    def money(self, value: Decimal) -> str:
        return f"{self.currency}{self.num(value)}"

    def date(self, d: date) -> str:
        return d.strftime(self.date_format)


def qty(value: Decimal) -> str:
    return f"{value.normalize():f}" if value == value.to_integral() else f"{value:f}".rstrip("0")


def gstin_or(party: Party, fallback: str) -> str:
    return party.gstin or fallback


def tax_lines(draft: Draft, s: Style, width: int, label_style: str) -> list[str]:
    out = []
    rates = draft.tax_by_rate()
    single = len(rates) == 1
    rate = next(iter(rates))
    names = (
        (("CGST", draft.cgst), ("SGST", draft.sgst))
        if draft.intra_state
        else (("IGST", draft.igst),)
    )
    for name, amount in names:
        pct = rate / 2 if name != "IGST" else rate
        pct_text = f"{pct:g}%"
        if label_style == "at" and single:
            label = f"{name} @ {pct_text}"
        elif label_style == "paren" and single:
            label = f"{name} ({pct_text})"
        else:
            label = name
        out.append(f"{L(label)}:".rjust(width - 14 + 2) + s.money(amount).rjust(14))
    return out


def render_classic(draft: Draft, rng: random.Random) -> tuple[str, Invoice]:
    s = Style.random(rng)
    w = 78
    buyer_label = rng.choice(["Bill To", "Billed To", "Buyer", "Customer Details", "Sold To"])
    total_label = rng.choice(["Grand Total", "Total", "Invoice Total", "Total Amount"])
    taxable_label = rng.choice(["Taxable Value", "Sub Total", "Taxable Amount", "Total Before Tax"])
    lines = [L("TAX INVOICE").center(w), "", draft.seller.name, *draft.seller.address]
    lines.append(
        f"{L('GSTIN')}: {draft.seller.gstin}    {L('State')}: {draft.seller.state.name} ({draft.seller.state.code})"
    )
    if draft.seller.phone:
        lines.append(f"{L('Phone')}: {draft.seller.phone}")
    lines += ["-" * w]
    no_label = rng.choice(["Invoice No", "Invoice No.", "Invoice #", "Inv. No"])
    lines.append(
        f"{L(no_label)}: {draft.invoice_number}".ljust(44)
        + f"{L('Date')}: {s.date(draft.invoice_date)}"
    )
    pos = rng.choice(
        [
            f"{draft.place_of_supply.code}-{draft.place_of_supply.name}",
            f"{draft.place_of_supply.name} ({draft.place_of_supply.code})",
            draft.place_of_supply.name,
        ]
    )
    lines.append(f"{L('Place of Supply')}: {pos}")
    lines += ["", f"{L(buyer_label)}:", draft.buyer.name, *draft.buyer.address]
    lines.append(
        f"{L('GSTIN')}: {gstin_or(draft.buyer, rng.choice(['Unregistered', 'URP', 'N/A', '-']))}"
    )
    lines += ["-" * w]
    header = f"{'#':<3}{'Description':<34}{'HSN/SAC':<9}{'Qty':>7}{'Rate':>12}{'Amount':>13}"
    lines += [L(header), "-" * w]
    for i, r in enumerate(draft.rows, 1):
        lines.append(
            f"{i:<3}{r.description[:33]:<34}{r.hsn:<9}{qty(r.quantity):>7}{s.num(r.rate):>12}{s.num(r.amount):>13}"
        )
    lines += ["-" * w]
    lines.append(f"{L(taxable_label)}:".rjust(w - 14 + 2) + s.money(draft.taxable_value).rjust(14))
    lines += tax_lines(draft, s, w, rng.choice(["at", "paren", "plain"]))
    if draft.round_off:
        lines.append(f"{L('Round Off')}:".rjust(w - 14 + 2) + s.num(draft.round_off).rjust(14))
    lines.append(f"{L(total_label)}:".rjust(w - 14 + 2) + s.money(draft.total).rjust(14))
    if rng.random() < 0.6:
        lines += ["", f"{L('Amount in words')}: {rupees_in_words(draft.total)}"]
    lines += ["", f"For {draft.seller.name}".rjust(w), L("Authorised Signatory").rjust(w)]
    return "\n".join(lines), draft.to_invoice()


def render_tally(draft: Draft, rng: random.Random) -> tuple[str, Invoice]:
    s = Style.random(rng)
    s = replace(s, grouping="indian" if rng.random() < 0.8 else s.grouping)
    lines = [L("Tax Invoice").center(80), "", draft.seller.name, *draft.seller.address]
    lines.append(f"{L('GSTIN/UIN')}: {draft.seller.gstin}")
    lines.append(
        f"{L('State Name')} : {draft.seller.state.name}, {L('Code')} : {draft.seller.state.code}"
    )
    lines += [
        "",
        f"{L('Invoice No.'):<30}{L('Dated')}",
        f"{draft.invoice_number:<28}{s.date(draft.invoice_date)}",
        "",
    ]
    if rng.random() < 0.5:
        lines += [L("Consignee (Ship to)"), draft.buyer.name, *draft.buyer.address, ""]
    lines += [L("Buyer (Bill to)"), draft.buyer.name, *draft.buyer.address]
    if draft.buyer.gstin:
        lines.append(f"{L('GSTIN/UIN')}       : {draft.buyer.gstin}")
    lines.append(
        f"{L('State Name')}      : {draft.buyer.state.name}, {L('Code')} : {draft.buyer.state.code}"
    )
    lines.append(f"{L('Place of Supply')} : {draft.place_of_supply.name}")
    lines += [
        "",
        L(
            f"{'Sl':<4}{'Description of Goods':<34}{'HSN/SAC':<9}{'Quantity':>12}{'Rate':>11} {'per':<6}{'Amount':>13}"
        ),
    ]
    for i, r in enumerate(draft.rows, 1):
        lines.append(
            f"{i:<4}{r.description[:33]:<34}{r.hsn:<9}{qty(r.quantity) + ' ' + r.unit:>12}"
            f"{s.num(r.rate):>11} {r.unit:<6}{s.num(r.amount):>13}"
        )
    lines.append(" " * 76 + "-" * 13)
    lines.append(f"{s.num(draft.taxable_value):>89}")
    for rate, base in draft.tax_by_rate().items():
        if draft.intra_state:
            half = draft.half_tax(rate, base)
            lines.append(f"{'':<4}{L(f'Output CGST @ {rate / 2:g}%'):<79}{s.num(half):>13}")
            lines.append(f"{'':<4}{L(f'Output SGST @ {rate / 2:g}%'):<79}{s.num(half):>13}")
        else:
            lines.append(
                f"{'':<4}{L(f'Output IGST @ {rate}%'):<79}{s.num(draft.full_tax(rate, base)):>13}"
            )
    if draft.round_off:
        ro = draft.round_off
        ro_text = f"(-){s.num(-ro)}" if ro < 0 else s.num(ro)
        lines.append(f"{'':<4}{L('Round Off'):<79}{ro_text:>13}")
    lines.append(" " * 76 + "-" * 13)
    lines.append(f"{L('Total'):<62}{s.money(draft.total):>29}")
    lines += [
        L("Amount Chargeable (in words)"),
        f"INR {rupees_in_words(draft.total).removeprefix('Rupees ')}",
    ]
    lines += [
        "",
        L("Declaration"),
        "We declare that this invoice shows the actual price of the goods",
        "described and that all particulars are true and correct.",
    ]
    lines += [f"for {draft.seller.name}".rjust(80), L("Authorised Signatory").rjust(80)]
    return "\n".join(lines), draft.to_invoice()


def render_key_value(draft: Draft, rng: random.Random) -> tuple[str, Invoice]:
    s = Style.random(rng)
    seller_kw = rng.choice(["Supplier", "Seller", "Vendor", "From"])
    buyer_kw = rng.choice(["Customer", "Buyer", "Recipient", "To"])
    header = [
        f"{L('Invoice Number')}: {draft.invoice_number}",
        f"{L('Invoice Date')}: {s.date(draft.invoice_date)}",
    ]
    seller = [
        f"{L(seller_kw)}: {draft.seller.name}",
        f"{L(seller_kw + ' GSTIN')}: {draft.seller.gstin}",
        f"{L(seller_kw + ' Address')}: {', '.join(draft.seller.address)}",
    ]
    buyer = [f"{L(buyer_kw)}: {draft.buyer.name}"]
    if draft.buyer.gstin or rng.random() < 0.5:
        buyer.append(f"{L(buyer_kw + ' GSTIN')}: {gstin_or(draft.buyer, 'Not Registered')}")
    buyer.append(f"{L(buyer_kw + ' Address')}: {', '.join(draft.buyer.address)}")
    pos = [f"{L('Place of Supply')}: {draft.place_of_supply.name} ({draft.place_of_supply.code})"]
    blocks = [header, seller, buyer, pos]
    rng.shuffle(blocks)
    lines = [line for block in blocks for line in block]
    lines.append(f"{L('Items')}:")
    for i, r in enumerate(draft.rows, 1):
        lines.append(
            f"{i}. {r.description} | {L('HSN')} {r.hsn} | {L('Qty')} {qty(r.quantity)} {r.unit} | "
            f"{L('Rate')} {s.num(r.rate)} | {L('Amt')} {s.num(r.amount)}"
        )
    lines.append(f"{L('Taxable Amount')}: {s.money(draft.taxable_value)}")
    lines.append(f"{L('CGST')}: {s.money(draft.cgst)}")
    lines.append(f"{L('SGST')}: {s.money(draft.sgst)}")
    lines.append(f"{L('IGST')}: {s.money(draft.igst)}")
    if draft.round_off:
        lines.append(f"{L('Rounding')}: {s.num(draft.round_off)}")
    lines.append(
        f"{L(rng.choice(['Invoice Total', 'Total Payable', 'Net Amount']))}: {s.money(draft.total)}"
    )
    return "\n".join(lines), draft.to_invoice()


def render_thermal(draft: Draft, rng: random.Random) -> tuple[str, Invoice]:
    s = replace(Style.random(rng), grouping="plain", currency="")
    s = replace(s, date_format=rng.choice(["%d/%m/%y", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%y"]))
    w = 44
    seller = replace(draft.seller, name=draft.seller.name.upper())
    buyer = replace(draft.buyer, name=draft.buyer.name.upper())
    rows = [replace(r, description=r.description.upper()[:16].rstrip()) for r in draft.rows]
    d = replace(draft, seller=seller, buyer=buyer, rows=rows)
    lines = [d.seller.name.center(w), *(a.upper().center(w) for a in d.seller.address)]
    lines.append(f"{L('GSTIN')}: {d.seller.gstin}".center(w + 2))
    if d.seller.phone:
        lines.append(f"{L('PH')}: {d.seller.phone}".center(w + 2))
    lines += [L("TAX INVOICE").center(w + 2), "-" * w]
    lines.append(
        f"{L('BILL NO')}: {d.invoice_number}".ljust(26) + f"{L('DT')}: {s.date(d.invoice_date)}"
    )
    lines.append(f"{L('CUST')}: {d.buyer.name}")
    if d.buyer.gstin:
        lines.append(f"{L('CUST GSTIN')}: {d.buyer.gstin}")
    lines.append(f"{L('POS')}: {d.place_of_supply.name.upper()}")
    lines += ["-" * w, L(f"{'ITEM':<17}{'HSN':<7}{'QTY':>5}{'RATE':>7}{'AMT':>8}"), "-" * w]
    for r in d.rows:
        rate_text = s.num(r.rate)
        amt_text = s.num(r.amount)
        if len(rate_text) + len(amt_text) > 14:
            lines.append(f"{r.description:<17}{r.hsn:<7}{qty(r.quantity):>5}")
            lines.append(f"{'':<10}{rate_text:>16}{amt_text:>18}")
        else:
            lines.append(
                f"{r.description:<17}{r.hsn:<7}{qty(r.quantity):>5}{rate_text:>7}{amt_text:>8}"
            )
    lines.append("-" * w)
    lines.append(f"{L('TAXABLE AMT'):<28}{s.num(d.taxable_value):>18}")
    if d.intra_state:
        lines.append(f"{L('CGST'):<28}{s.num(d.cgst):>18}")
        lines.append(f"{L('SGST'):<28}{s.num(d.sgst):>18}")
    else:
        lines.append(f"{L('IGST'):<28}{s.num(d.igst):>18}")
    if d.round_off:
        lines.append(f"{L('ROUND OFF'):<28}{s.num(d.round_off):>18}")
    lines.append(f"{L('NET AMOUNT'):<28}{s.num(d.total):>18}")
    lines += ["-" * w, L("THANK YOU! VISIT AGAIN").center(w + 2)]
    return "\n".join(lines), d.to_invoice()


def render_email(draft: Draft, rng: random.Random) -> tuple[str, Invoice]:
    s = Style.random(rng)
    s = replace(s, currency=rng.choice(["Rs. ", "INR ", "₹"]))
    contact = rng.choice(["Team", "Sir/Madam", "Accounts Team"])
    buyer_ref = draft.buyer.name
    if draft.buyer.gstin:
        buyer_ref += f" (GSTIN {draft.buyer.gstin})"
    lines = [
        f"{L('Subject')}: Invoice {draft.invoice_number} from {draft.seller.name}",
        "",
        f"Dear {contact},",
        "",
        f"Please find below the details of our invoice no. {draft.invoice_number} dated "
        f"{s.date(draft.invoice_date)} for supplies made to {buyer_ref}. "
        f"Place of supply is {draft.place_of_supply.name}.",
        "",
        f"{L('Items supplied')}:",
    ]
    for r in draft.rows:
        lines.append(
            f"- {qty(r.quantity)} x {r.description} ({L('HSN')} {r.hsn}) @ {s.money(r.rate)} = {s.money(r.amount)}"
        )
    lines += ["", f"{L('Taxable value')}: {s.money(draft.taxable_value)}"]
    if draft.intra_state:
        lines.append(f"{L('CGST')}: {s.money(draft.cgst)}, {L('SGST')}: {s.money(draft.sgst)}")
    else:
        lines.append(f"{L('IGST')}: {s.money(draft.igst)}")
    if draft.round_off:
        lines.append(f"{L('Round off')}: {s.num(draft.round_off)}")
    lines.append(f"{L('Total payable')}: {s.money(draft.total)}")
    lines += [
        "",
        "Kindly arrange the payment within 30 days.",
        "",
        "Regards,",
        rng.choice(["Accounts Department", "Billing Team", "Finance"]),
        draft.seller.name,
        f"{L('GSTIN')}: {draft.seller.gstin}",
        ", ".join(draft.seller.address),
    ]
    return "\n".join(lines), draft.to_invoice()


def _csv_cell(value: str) -> str:
    return f'"{value}"' if "," in value else value


def render_csv(draft: Draft, rng: random.Random) -> tuple[str, Invoice]:
    s = replace(Style.random(rng), grouping="plain", currency="")
    sep = rng.choice([",", ";", "\t"])

    def cell(value: str) -> str:
        return _csv_cell(value) if sep == "," else value

    def row(*cells: str) -> str:
        return sep.join(cell(c) for c in cells)

    lines = [
        row(L("Seller Name"), draft.seller.name),
        row(L("Seller GSTIN"), draft.seller.gstin or ""),
        row(L("Buyer Name"), draft.buyer.name),
        row(L("Buyer GSTIN"), draft.buyer.gstin or ""),
        row(L("Invoice No"), draft.invoice_number),
        row(L("Invoice Date"), s.date(draft.invoice_date)),
        row(L("Place Of Supply"), f"{draft.place_of_supply.code}-{draft.place_of_supply.name}"),
        "",
        row(*(L(h) for h in ("Sr", "Item", "HSN/SAC", "Qty", "Unit", "Rate", "GST %", "Amount"))),
    ]
    for i, r in enumerate(draft.rows, 1):
        lines.append(
            row(
                str(i),
                r.description,
                r.hsn,
                qty(r.quantity),
                r.unit,
                s.num(r.rate),
                str(r.gst_rate),
                s.num(r.amount),
            )
        )
    lines += [
        "",
        row(L("Taxable Value"), s.num(draft.taxable_value)),
        row(L("CGST"), s.num(draft.cgst)),
        row(L("SGST"), s.num(draft.sgst)),
        row(L("IGST"), s.num(draft.igst)),
    ]
    if draft.round_off:
        lines.append(row(L("Round Off"), s.num(draft.round_off)))
    lines.append(row(L("Invoice Value"), s.num(draft.total)))
    return "\n".join(lines), draft.to_invoice()


def render_einvoice(draft: Draft, rng: random.Random) -> tuple[str, Invoice]:
    """Layout modelled on the GST e-invoice portal print. Held out from training."""
    if draft.buyer.gstin is None:
        buyer = replace(
            draft.buyer, gstin=make_gstin(draft.buyer.state.code, draft.buyer.name, rng)
        )
        draft = replace(draft, buyer=buyer)
    s = replace(Style.random(rng), grouping=rng.choice(["plain", "indian"]), currency="")
    s = replace(s, date_format=rng.choice(["%d/%m/%Y", "%d-%m-%Y", "%d-%b-%Y"]))
    irn = "".join(rng.choices("0123456789abcdef", k=64))
    ack_no = str(rng.randint(10**14, 10**15 - 1))
    ack_date = (
        draft.invoice_date.strftime("%d/%m/%Y")
        + f" {rng.randint(9, 20):02d}:{rng.randint(0, 59):02d}:00"
    )
    col = 46
    seller_lines = [
        f"{L('GSTIN')} : {draft.seller.gstin}",
        f"{L('Legal Name')} : {draft.seller.name}",
        *draft.seller.address,
    ]
    buyer_lines = [
        f"{L('GSTIN')} : {draft.buyer.gstin or 'URP'}",
        f"{L('Legal Name')} : {draft.buyer.name}",
        *draft.buyer.address,
        f"{L('Place of Supply')} : {draft.place_of_supply.name.upper()}",
    ]
    lines = [
        L("e-Invoice"),
        f"{L('IRN')} : {irn}",
        f"{L('Ack No.')} : {ack_no}    {L('Ack Date')} : {ack_date}",
        "",
        L("1. e-Invoice Details"),
        f"{L('Category')} : B2B    {L('Document Type')} : Tax Invoice    "
        f"{L('Document No')} : {draft.invoice_number}    {L('Document Date')} : {s.date(draft.invoice_date)}",
        "",
        L("2. Party Details"),
        f"{L('Supplier')} :".ljust(col + 2) + f"{L('Recipient')} :",
    ]
    for i in range(max(len(seller_lines), len(buyer_lines))):
        left = seller_lines[i] if i < len(seller_lines) else ""
        right = buyer_lines[i] if i < len(buyer_lines) else ""
        pad = col + 2 * left.count(LABEL_OPEN)
        lines.append(f"{left:<{pad}}{right}")
    lines += [
        "",
        L("3. Details of Goods / Services"),
        L(
            f"{'SlNo':<5}{'Item Description':<34}{'HSN Code':<10}{'Quantity':>9} {'Unit':<6}{'Unit Price(Rs)':>15}"
            f"{'Taxable Amount(Rs)':>20}{'Tax Rate':>10}{'Total(Rs)':>14}"
        ),
    ]
    for i, r in enumerate(draft.rows, 1):
        tax = draft.full_tax(r.gst_rate, r.amount)
        lines.append(
            f"{i:<5}{r.description[:33]:<34}{r.hsn:<10}{qty(r.quantity):>9} {r.unit.upper():<6}{s.num(r.rate):>15}"
            f"{s.num(r.amount):>20}{f'{r.gst_rate}%':>10}{s.num(r.amount + tax):>14}"
        )
    summary_heads = (
        "Tax'ble Amt",
        "CGST Amt",
        "SGST Amt",
        "IGST Amt",
        "CESS Amt",
        "Round off",
        "Total Inv. Amt",
    )
    summary_vals = (
        draft.taxable_value,
        draft.cgst,
        draft.sgst,
        draft.igst,
        Decimal("0"),
        draft.round_off,
        draft.total,
    )
    lines += [
        "",
        L("4. Tax Summary"),
        L("".join(f"{h:>15}" for h in summary_heads)),
        "".join(f"{s.num(v):>15}" for v in summary_vals),
        "",
        f"{L('Generated By')} : {draft.seller.gstin}",
        f"{L('Print Date')} : {draft.invoice_date.strftime('%d/%m/%Y')}",
    ]
    return "\n".join(lines), draft.to_invoice()


Renderer = Callable[[Draft, random.Random], tuple[str, Invoice]]

TRAIN_TEMPLATES: dict[str, Renderer] = {
    "classic": render_classic,
    "tally": render_tally,
    "key_value": render_key_value,
    "thermal": render_thermal,
    "email": render_email,
    "csv": render_csv,
}
HELDOUT_TEMPLATES: dict[str, Renderer] = {"einvoice": render_einvoice}
ALL_TEMPLATES: dict[str, Renderer] = {**TRAIN_TEMPLATES, **HELDOUT_TEMPLATES}

_OCR_SWAPS = {
    "o": "0",
    "O": "0",
    "l": "1",
    "I": "l",
    "S": "5",
    "e": "c",
    "i": "l",
    "B": "8",
    "a": "o",
}


def _corrupt_label(label: str, rng: random.Random) -> str:
    chars = []
    for ch in label:
        roll = rng.random()
        if roll < 0.04 and ch in _OCR_SWAPS:
            chars.append(_OCR_SWAPS[ch])
        elif roll < 0.052 and ch.isalpha():
            continue
        elif roll < 0.064 and ch != " ":
            chars.append(ch + " ")
        else:
            chars.append(ch)
    return "".join(chars)


def add_ocr_noise(text: str, rng: random.Random) -> str:
    text = LABEL_RE.sub(lambda m: _corrupt_label(m.group(1), rng), text)
    out = []
    for line in text.split("\n"):
        roll = rng.random()
        if roll < 0.15:
            line = re.sub(
                r" {2,}", lambda m: " " * max(1, len(m.group(0)) + rng.randint(-3, 3)), line
            )
        elif roll < 0.2:
            line = line.replace(": ", rng.choice([" : ", ":", "; ", ". "]), 1)
        elif roll < 0.23:
            line = line + rng.choice([" |", " _", " .", " ,"])
        if rng.random() < 0.04:
            out.append("")
        if rng.random() < 0.05:
            line = " " * rng.randint(1, 6) + line
        out.append(line)
    return "\n".join(out)


def strip_labels(text: str) -> str:
    return LABEL_RE.sub(lambda m: m.group(1), text)


def _rows_fit(rows: list[Row]) -> bool:
    return all(len(r.description) <= 33 for r in rows)


def render(template: str, draft: Draft, rng: random.Random, ocr_noise: bool) -> tuple[str, Invoice]:
    assert _rows_fit(draft.rows)
    text, invoice = ALL_TEMPLATES[template](draft, rng)
    text = add_ocr_noise(text, rng) if ocr_noise else strip_labels(text)
    return text.rstrip() + "\n", invoice
