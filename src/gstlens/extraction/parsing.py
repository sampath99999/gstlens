import json
import re
from dataclasses import dataclass
from typing import Any

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


@dataclass(frozen=True)
class ParsedOutput:
    data: dict[str, Any] | None
    raw_valid: bool
    repaired: bool
    error: str | None = None


def _close_open_brackets(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if in_string:
        text += '"'
    text = re.sub(r",\s*$", "", text.rstrip())
    return text + "".join(reversed(stack))


def _try_load(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_model_output(raw: str) -> ParsedOutput:
    """Parse the model reply as JSON, repairing common formatting slips.

    raw_valid reports whether the untouched reply was already a JSON object.
    """
    stripped = raw.strip()
    data = _try_load(stripped)
    if data is not None:
        return ParsedOutput(data, raw_valid=True, repaired=False)

    candidate = FENCE_RE.sub("", stripped)
    start = candidate.find("{")
    if start == -1:
        return ParsedOutput(None, raw_valid=False, repaired=False, error="no JSON object found")
    end = candidate.rfind("}")
    attempts = []
    if end > start:
        attempts.append(candidate[start : end + 1])
    attempts.append(_close_open_brackets(candidate[start:]))
    for attempt in attempts:
        for text in (attempt, TRAILING_COMMA_RE.sub(r"\1", attempt)):
            data = _try_load(text)
            if data is not None:
                return ParsedOutput(data, raw_valid=False, repaired=True)
    return ParsedOutput(None, raw_valid=False, repaired=False, error="could not repair JSON")
