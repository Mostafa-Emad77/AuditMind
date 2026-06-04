"""Canonical ID determinism + dedup semantics."""
from app.utils.canonical_id import (
    canonical_entity_id,
    canonical_rel_id,
    _GLOBAL_DEDUPE_TYPES,
)


def test_global_types_collapse_across_docs():
    """A company / contract_id mentioned in two docs gets ONE entity_id."""
    for etype in _GLOBAL_DEDUPE_TYPES:
        a = canonical_entity_id(etype, "Delta Trading S.A.E.", "doc-A")
        b = canonical_entity_id(etype, "Delta Trading S.A.E.", "doc-B")
        assert a == b, f"{etype} should be doc-agnostic"


def test_local_types_are_doc_scoped():
    """Amounts / dates / clauses must NOT collapse across docs."""
    for etype in ("amount", "date", "clause", "other"):
        a = canonical_entity_id(etype, "50000 EGP", "doc-A")
        b = canonical_entity_id(etype, "50000 EGP", "doc-B")
        assert a != b, f"{etype} must stay doc-scoped"


def test_normalization_is_case_and_whitespace_insensitive():
    a = canonical_entity_id("company", "Delta Trading", "doc-A")
    b = canonical_entity_id("company", "  delta trading  ", "doc-B")
    assert a == b


def test_distinct_values_get_distinct_ids():
    a = canonical_entity_id("company", "Delta Trading", "doc-A")
    b = canonical_entity_id("company", "Delta Trading S.A.E.", "doc-A")
    # Legal-suffix preservation: different normalized_value -> different id.
    assert a != b


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
