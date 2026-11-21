import random

import pytest

from gstlens.gstin import CHARSET, checksum_char, is_valid_gstin, make_gstin


@pytest.mark.parametrize("gstin", ["27AAPFU0939F1ZV", "29AAGCB7383J1Z4"])
def test_known_gstins_are_valid(gstin: str) -> None:
    assert is_valid_gstin(gstin)


def test_single_character_typo_is_caught() -> None:
    gstin = "27AAPFU0939F1ZV"
    for i in range(14):
        original = gstin[i]
        pool = "0123456789" if original.isdigit() else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for ch in pool:
            if ch == original:
                continue
            mutated = gstin[:i] + ch + gstin[i + 1 :]
            assert not is_valid_gstin(mutated), mutated


def test_rejects_bad_format() -> None:
    assert not is_valid_gstin("27AAPFU0939F1V")
    assert not is_valid_gstin("27AAPFU0939F1XV")
    assert not is_valid_gstin("")


def test_generated_gstins_are_valid_and_carry_state_code() -> None:
    rng = random.Random(0)
    for _ in range(500):
        code = f"{rng.randint(1, 37):02d}"
        gstin = make_gstin(code, "Kaveri Traders", rng)
        assert is_valid_gstin(gstin)
        assert gstin.startswith(code)
        assert gstin[13] == "Z"
        assert gstin[6] == "K"


def test_checksum_is_in_charset() -> None:
    assert checksum_char("27AAPFU0939F1Z") in CHARSET
