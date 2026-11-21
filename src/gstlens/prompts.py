import json

from gstlens.schema import SCHEMA_HINT

SYSTEM_PROMPT = (
    "You extract structured data from Indian GST invoices. "
    "Reply with a single JSON object and nothing else, using exactly these keys:\n"
    f"{SCHEMA_HINT}\n"
    "Amounts are plain numbers. Item amount is before tax. Use 0 for taxes that do not apply "
    "and null for a missing buyer GSTIN. place_of_supply is the state name."
)


def target_to_text(target: dict) -> str:
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def build_messages(text: str, few_shot: list[dict] | None = None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in few_shot or []:
        messages.append({"role": "user", "content": example["text"]})
        messages.append({"role": "assistant", "content": target_to_text(example["target"])})
    messages.append({"role": "user", "content": text})
    return messages
