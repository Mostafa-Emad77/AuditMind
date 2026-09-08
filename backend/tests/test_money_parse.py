"""Unit tests for deterministic monetary parsing."""

from app.utils.money_parse import parse_monetary_amount, amounts_within_tiny_tolerance


def test_plain_currency_amounts():
    assert parse_monetary_amount("50000 EGP") == 50000.0
    assert parse_monetary_amount("EGP 50,000.00") == 50000.0
    assert parse_monetary_amount("USD 1,234.56") == 1234.56


def test_arabic_indic_digits():
    assert parse_monetary_amount("٥٠٠٠٠ جنيه") == 50000.0


def test_filename_and_id_noise_does_not_concatenate():
    # Old bug: stripping non-digits concatenated 112+2024+50000 etc.
    assert parse_monetary_amount("contract_112_2024_50000") is None
    assert parse_monetary_amount("50000 (from invoice_2024_991.pdf)") is None


def test_decorated_string_without_raw_amounts_is_ambiguous():
    s = "{50000} (from contract_2024.pdf)"
    assert parse_monetary_amount(s) is None


def test_single_bare_number():
    assert parse_monetary_amount("50000") == 50000.0


def test_multiple_disparate_amounts_rejected():
    assert parse_monetary_amount("100 EGP 200 USD") is None


def test_year_token_skipped_when_currency_present():
    # If only year-like run with currency elsewhere — currency path should pick the money
    v = parse_monetary_amount("2024 SAR 5000")
    assert v == 5000.0


def test_tiny_tolerance():
    assert amounts_within_tiny_tolerance(100.0, 100.000001)
    assert amounts_within_tiny_tolerance(1e9, 1e9 + 0.005)
