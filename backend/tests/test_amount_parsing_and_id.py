"""Amount parsing → canonical id collapse + currency extraction + chunk point IDs."""
from app.utils.canonical_id import (
    canonical_chunk_point_id,
    canonical_entity_id,
    canonical_rel_id,
)
from app.utils.money_parse import (
    parse_monetary_amount,
    extract_currency_code,
    amounts_within_tiny_tolerance,
)


def test_currency_extraction_basic():
    assert extract_currency_code("50,000 EGP") == "EGP"
    assert extract_currency_code("USD 1,200.50") == "USD"
    assert extract_currency_code("EUR 999") == "EUR"


def test_currency_extraction_arabic():
    assert extract_currency_code("١٠٠٠ جنيه") == "EGP"
    assert extract_currency_code("500 ريال") == "SAR"


def test_currency_extraction_none_when_absent_or_ambiguous():
    assert extract_currency_code("plain 1000") is None
    # Two different currencies → ambiguous
    assert extract_currency_code("USD 100 EGP 200") is None


def test_amount_canonical_id_collapses_equivalent_strings_in_doc():
    """50,000 EGP == 50000 EGP == EGP 50000.00 within the same doc."""
    base = ("amount", "ignored", "doc-A")
    a = canonical_entity_id(*base, amount_value=50000.0, amount_currency="EGP")
    b = canonical_entity_id(*base, amount_value=50000.00, amount_currency="EGP")
    c = canonical_entity_id(*base, amount_value=50000.001, amount_currency="EGP")  # rounds to 50000.00
    assert a == b == c


def test_amount_canonical_id_separates_currencies():
    a = canonical_entity_id("amount", "x", "doc-A", amount_value=100.0, amount_currency="USD")
    b = canonical_entity_id("amount", "x", "doc-A", amount_value=100.0, amount_currency="EGP")
    assert a != b


def test_amount_canonical_id_doc_scoped():
    """Same amount in two docs is two separate entity nodes."""
    a = canonical_entity_id("amount", "x", "doc-A", amount_value=100.0, amount_currency="USD")
    b = canonical_entity_id("amount", "x", "doc-B", amount_value=100.0, amount_currency="USD")
    assert a != b


def test_amount_canonical_id_falls_back_when_unparsed():
    """If amount_value is None we still get a stable id from normalized_value."""
    a = canonical_entity_id("amount", "fifty thousand pounds", "doc-A")
    b = canonical_entity_id("amount", "fifty thousand pounds", "doc-A")
    assert a == b


def test_chunk_point_id_deterministic_and_index_sensitive():
    a = canonical_chunk_point_id("doc-A", 0)
    b = canonical_chunk_point_id("doc-A", 0)
    c = canonical_chunk_point_id("doc-A", 1)
    d = canonical_chunk_point_id("doc-B", 0)
    assert a == b
    assert a != c
    assert a != d


def test_amounts_within_tolerance_rounding_drift():
    assert amounts_within_tiny_tolerance(100.00, 100.001)
    assert not amounts_within_tiny_tolerance(100.0, 130.0)


def test_parse_monetary_amount_roundtrip():
    assert parse_monetary_amount("50,000 EGP") == 50000.0
    assert parse_monetary_amount("EGP 50000.00") == 50000.0
    assert parse_monetary_amount("١٠٠ جنيه") == 100.0


def test_rel_id_deterministic_and_directional():
    r1 = canonical_rel_id("e1", "e2", "total_value_of")
    r2 = canonical_rel_id("e1", "e2", "total_value_of")
    assert r1 == r2
    # Direction matters
    assert r1 != canonical_rel_id("e2", "e1", "total_value_of")
    # Type matters
    assert r1 != canonical_rel_id("e1", "e2", "payment_for")


def test_rel_type_case_insensitive():
    a = canonical_rel_id("x", "y", "Payment_For")
    b = canonical_rel_id("x", "y", "payment_for")
    assert a == b
