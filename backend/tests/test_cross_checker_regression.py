"""Regression: graph-style amounts must not produce fabricated billion-level discrepancies."""
import json

import pytest

from app.agents.cross_checker import (
    _evidence_line_backs_value,
    _evidence_snippets_back_values,
    _graph_context_for_compare,
)
from app.tools.financial import compare_values


def test_compare_values_raw_only_no_filename_leak():
    """value1/value2 must be raw monetary strings; context holds provenance."""
    raw = compare_values.invoke({
        "value1": "50000 EGP",
        "value2": "45000 EGP",
        "context": "from Invoice A.pdf vs Contract B.pdf",
    })
    data = json.loads(raw)
    assert data["is_contradiction"] is True
    assert "billion" not in data["explanation"].lower()
    # No concatenated monster number in output
    assert "5000045000" not in json.dumps(data)


def test_parse_failure_skips_numeric_contradiction():
    raw = compare_values.invoke({
        "value1": "maybe not a number",
        "value2": "also vague",
        "context": "test",
    })
    data = json.loads(raw)
    assert data["is_contradiction"] is False
    assert data["severity"] == "ok"


def test_evidence_backs_typical_snippet():
    assert _evidence_line_backs_value("Contract.pdf, page 1: 50000 EGP", "50000 EGP")
    assert not _evidence_line_backs_value("Contract.pdf, page 1: unrelated", "50000 EGP")


def test_evidence_snippets_back_values_for_bank_recon():
    ev = [
        "contract.pdf, page 1: Article 3 contract value EGP 200,000",
        "bank.pdf, page 1: Payment transfer EGP 226,700",
    ]
    assert _evidence_snippets_back_values(ev, 200000.0, 226700.0)


def test_bank_recon_detectors_return_lists_not_hallucinated_amounts():
    """Sanity: detectors return structured dicts; no string concatenation of amounts."""
    from app.agents.cross_checker import detect_bank_overpayment_gaps
    from app.models.schemas import DocumentMeta

    docs = [
        DocumentMeta(doc_id="c", filename="c.pdf", doc_type="contract"),
        DocumentMeta(doc_id="b", filename="b.pdf", doc_type="bank_statement"),
    ]
    results = [
        {"doc_id": "c", "page": 1, "text": "Contract value EGP 100,000."},
        {"doc_id": "b", "page": 1, "text": "Payment transfer EGP 100,001."},
    ]
    gaps = detect_bank_overpayment_gaps(results, docs)
    dumped = json.dumps(gaps)
    assert "100001100000" not in dumped


def test_graph_context_includes_anchor():
    ctx = _graph_context_for_compare(
        {
            "page1": 1,
            "page2": 2,
            "anchor_type": "contract_id",
            "anchor_value": "C-99",
            "anchor_entity_id": "e-anchor",
        },
        "A.pdf",
        "B.pdf",
    )
    assert "C-99" in ctx
    assert "anchor" in ctx.lower()


@pytest.mark.parametrize("v1,v2,expect_contra", [
    ("100.00 EGP", "100.01 EGP", False),  # tiny drift → tolerance
    ("100 EGP", "130 EGP", True),
])
def test_real_mismatch_vs_rounding(v1, v2, expect_contra):
    raw = compare_values.invoke({"value1": v1, "value2": v2, "context": ""})
    data = json.loads(raw)
    assert data["is_contradiction"] is expect_contra


def test_finding_dedup_key_ignores_evidence_and_txn_lines():
    """Same bank-vs-contract story must share a key despite different evidence / TXN amounts."""
    from app.agents.cross_checker import _finding_dedup_key
    from app.models.schemas import Finding

    a = Finding(
        severity="critical",
        title="Total payments exceed contract",
        description=(
            "Contract EGP 200,000. Bank paid EGP 226,700. Overpayment EGP 26,700. "
            "TXN-005 EGP 68,400."
        ),
        confidence_score=0.95,
        evidence=["bank.pdf, page 2: 326,000.00 374,300.00"],
        source_doc_id="c",
        conflicting_doc_id="b",
    )
    b = Finding(
        severity="critical",
        title="Bank paid more than contract value",
        description="EGP 226,700 vs EGP 200,000 — gap EGP 26,700.",
        confidence_score=0.92,
        evidence=["bank.pdf, page 2: 750,000.00 balance"],
        source_doc_id="c",
        conflicting_doc_id="b",
    )
    assert _finding_dedup_key(a) == _finding_dedup_key(b)
