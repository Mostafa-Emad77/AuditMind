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


def test_dedupe_findings_merges_same_bank_contract_story_despite_different_evidence():
    """Same bank-vs-contract story must dedupe to one despite different evidence / TXN lines."""
    from app.agents.cross_checker.dedup import dedupe_findings
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
    deduped = dedupe_findings([a, b])
    assert len(deduped) == 1
    )
    assert _finding_dedup_key(a) == _finding_dedup_key(b)
