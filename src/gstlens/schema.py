from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gstlens.gstin import GSTIN_RE


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    hsn: str = Field(pattern=r"^\d{4,8}$")
    quantity: float = Field(gt=0)
    rate: float = Field(ge=0)
    amount: float = Field(ge=0)


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seller_name: str = Field(min_length=1)
    seller_gstin: str
    buyer_name: str = Field(min_length=1)
    buyer_gstin: str | None = None
    invoice_number: str = Field(min_length=1)
    invoice_date: date
    place_of_supply: str = Field(min_length=1)
    line_items: list[LineItem] = Field(min_length=1)
    taxable_value: float = Field(ge=0)
    cgst: float = Field(ge=0)
    sgst: float = Field(ge=0)
    igst: float = Field(ge=0)
    total: float = Field(ge=0)

    @field_validator("seller_gstin", "buyer_gstin")
    @classmethod
    def _gstin_format(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().upper()
        if not GSTIN_RE.match(value):
            raise ValueError(
                "GSTIN must be 15 characters: 2-digit state code, PAN, entity, Z, check"
            )
        return value

    def to_target_json(self) -> dict:
        return self.model_dump(mode="json")


SCHEMA_HINT = (
    '{"seller_name": str, "seller_gstin": str, "buyer_name": str, "buyer_gstin": str|null, '
    '"invoice_number": str, "invoice_date": "YYYY-MM-DD", "place_of_supply": str, '
    '"line_items": [{"description": str, "hsn": str, "quantity": num, "rate": num, "amount": num}], '
    '"taxable_value": num, "cgst": num, "sgst": num, "igst": num, "total": num}'
)
