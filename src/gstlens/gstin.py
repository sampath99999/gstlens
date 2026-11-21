import random
import re
import string

CHARSET = string.digits + string.ascii_uppercase
GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")

# Fourth PAN character encodes the holder type.
PAN_HOLDER_TYPES = "CPFHAT"


def checksum_char(first14: str) -> str:
    total = 0
    for i, ch in enumerate(first14):
        product = CHARSET.index(ch) * (2 if i % 2 else 1)
        total += product // 36 + product % 36
    return CHARSET[(36 - total % 36) % 36]


def is_valid_gstin(value: str) -> bool:
    value = value.strip().upper()
    return bool(GSTIN_RE.match(value)) and checksum_char(value[:14]) == value[14]


def make_pan(name: str, rng: random.Random) -> str:
    first_letter = next((c for c in name.upper() if c.isalpha()), "A")
    head = "".join(rng.choices(string.ascii_uppercase, k=3))
    holder = rng.choice(PAN_HOLDER_TYPES)
    digits = f"{rng.randint(0, 9999):04d}"
    return f"{head}{holder}{first_letter}{digits}{rng.choice(string.ascii_uppercase)}"


def make_gstin(state_code: str, name: str, rng: random.Random) -> str:
    entity = rng.choice("1111111223456789A")
    body = f"{state_code}{make_pan(name, rng)}{entity}Z"
    return body + checksum_char(body)
