from gstlens.extraction.parsing import parse_model_output

FENCE = "`" * 3


def test_plain_json_is_raw_valid() -> None:
    parsed = parse_model_output('  {"a": 1}\n')
    assert parsed.raw_valid and not parsed.repaired
    assert parsed.data == {"a": 1}


def test_code_fence_is_repaired() -> None:
    parsed = parse_model_output(f'{FENCE}json\n{{"a": 1}}\n{FENCE}')
    assert not parsed.raw_valid and parsed.repaired
    assert parsed.data == {"a": 1}


def test_leading_prose_and_trailing_comma() -> None:
    parsed = parse_model_output('Here is the JSON: {"a": [1, 2,], "b": "x",}')
    assert parsed.data == {"a": [1, 2], "b": "x"}


def test_truncated_output_is_closed() -> None:
    parsed = parse_model_output('{"a": {"b": [1, 2')
    assert parsed.data == {"a": {"b": [1, 2]}}


def test_truncated_inside_string() -> None:
    parsed = parse_model_output('{"name": "Sri Bal')
    assert parsed.data == {"name": "Sri Bal"}


def test_garbage_is_reported() -> None:
    parsed = parse_model_output("I cannot help with that")
    assert parsed.data is None
    assert parsed.error == "no JSON object found"


def test_top_level_array_is_not_accepted() -> None:
    assert parse_model_output("[1, 2]").data is None
