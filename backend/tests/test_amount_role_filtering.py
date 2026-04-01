"""Tests for amount-role-aware filtering that prevents false-positive contradictions."""
import json

import pytest

from app.agents.cross_checker import (
    _classify_amount_role_from_context,
    detect_bank_overpayment_gaps,
    detect_invoice_bank_total_gaps,
    detect_milestone_payment_mismatch,
)
from app.models.schemas import DocumentMeta, Entity
from app.services.graph_builder import _roles_are_comparable
from app.tools.planner_tools import generate_checklist


# ── _roles_are_comparable ────────────────────────────────────────────────────

class TestRolesAreComparable:
    def test_opening_balance_vs_anything_is_incompatible(self):
        assert not _roles_are_comparable("opening_balance", "total_contract_value")
        assert not _roles_are_comparable("opening_balance", "total_invoice")
        assert not _roles_are_comparable("opening_balance", "single_payment")
        assert not _roles_are_comparable("opening_balance", "unknown")

    def test_closing_balance_vs_anything_is_incompatible(self):
        assert not _roles_are_comparable("closing_balance", "total_contract_value")
        assert not _roles_are_comparable("closing_balance", "single_payment")

    def test_total_contract_vs_total_invoice_is_compatible(self):
        assert _roles_are_comparable("total_contract_value", "total_invoice")
        assert _roles_are_comparable("total_invoice", "total_contract_value")

    def test_single_payment_vs_total_contract_is_incompatible(self):
        assert not _roles_are_comparable("single_payment", "total_contract_value")
        assert not _roles_are_comparable("total_contract_value", "single_payment")

    def test_single_payment_vs_total_invoice_is_incompatible(self):
        assert not _roles_are_comparable("single_payment", "total_invoice")

    def test_unknown_vs_anything_passes(self):
        assert _roles_are_comparable("unknown", "total_contract_value")
        assert _roles_are_comparable("unknown", "single_payment")
        assert _roles_are_comparable("unknown", "unknown")

    def test_milestone_vs_single_payment_is_compatible(self):
        assert _roles_are_comparable("milestone_scheduled", "single_payment")
        assert _roles_are_comparable("single_payment", "milestone_scheduled")

    def test_none_treated_as_unknown(self):
        assert _roles_are_comparable(None, "total_contract_value")
        assert _roles_are_comparable(None, None)


# ── _classify_amount_role_from_context ───────────────────────────────────────

class TestClassifyAmountRole:
    def test_opening_balance_line(self):
        role = _classify_amount_role_from_context(
            "Opening Balance: EGP 320,000.00", "bank_statement"
        )
        assert role == "opening_balance"

    def test_closing_balance_line(self):
        role = _classify_amount_role_from_context(
            "Closing Balance carried forward: EGP 93,300.00", "bank_statement"
        )
        assert role == "closing_balance"

    def test_bank_payment_transaction(self):
        role = _classify_amount_role_from_context(
            "Payment transfer to Nile Consulting EGP 50,000", "bank_statement"
        )
        assert role == "single_payment"

    def test_contract_value(self):
        role = _classify_amount_role_from_context(
            "Article 3: The total contract value is EGP 200,000", "contract"
        )
        assert role == "total_contract_value"

    def test_invoice_total(self):
        role = _classify_amount_role_from_context(
            "Invoice subtotal EGP 176,700 marked PAID", "invoice"
        )
        assert role == "total_invoice"

    def test_milestone_line(self):
        role = _classify_amount_role_from_context(
            "Milestone 1 due EGP 60,000", "contract"
        )
        assert role == "milestone_scheduled"

    def test_retainer_line(self):
        role = _classify_amount_role_from_context(
            "Monthly retainer EGP 50,000", "contract"
        )
        assert role == "retainer"

    def test_ambiguous_returns_unknown(self):
        role = _classify_amount_role_from_context(
            "Some amount EGP 10,000 mentioned", "unknown"
        )
        assert role == "unknown"


# ── False positive regression: opening balance vs invoice ────────────────────

class TestFalsePositiveRegression:
    """These reproduce the exact false-positive scenarios from production."""

    def test_opening_balance_not_compared_to_invoice_total(self):
        """320,000 opening balance must NOT be compared to 180,000 invoice claim."""
        assert not _roles_are_comparable("opening_balance", "total_invoice")

    def test_single_payment_not_compared_to_contract_total(self):
        """50,000 advance payment must NOT be compared to 200,000 contract total."""
        assert not _roles_are_comparable("single_payment", "total_contract_value")

    def test_context_classifier_catches_opening_balance(self):
        role = _classify_amount_role_from_context(
            "Opening Balance 320,000.00", "bank_statement"
        )
        assert role == "opening_balance"

    def test_context_classifier_catches_advance_payment(self):
        role = _classify_amount_role_from_context(
            "TXN-002 05/01/2025 Payment transfer to Nile Consulting 50,000",
            "bank_statement",
        )
        assert role == "single_payment"


# ── Real findings detection ──────────────────────────────────────────────────

class TestRealFindingsDetection:
    """Verify the detectors catch the real issues from the audit scenario."""

    DOCS = [
        DocumentMeta(doc_id="c1", filename="contract.pdf", doc_type="contract"),
        DocumentMeta(doc_id="b1", filename="bank_statement.pdf", doc_type="bank_statement"),
        DocumentMeta(doc_id="i1", filename="invoice.pdf", doc_type="invoice"),
    ]

    BANK_RESULTS = [
        {
            "doc_id": "b1",
            "page": 1,
            "text": (
                "Payment transfer EGP 50,000. Payment transfer EGP 68,400. "
                "Payment transfer EGP 62,930. Payment transfer EGP 45,370."
            ),
        },
    ]

    def test_overpayment_detected_bank_vs_contract(self):
        results = [
            {
                "doc_id": "c1",
                "page": 1,
                "text": "Article 3 contract value EGP 200,000 for the full engagement.",
            },
            *self.BANK_RESULTS,
        ]
        gaps = detect_bank_overpayment_gaps(results, self.DOCS)
        assert gaps, "Overpayment should be detected"
        g = gaps[0]
        assert abs(g["bank_total"] - 226700) < 1
        assert abs(g["contract_total"] - 200000) < 1
        assert abs(g["variance"] - 26700) < 1
        assert g["severity"] == "critical"

    def test_bank_vs_invoice_mismatch_detected(self):
        results = [
            {
                "doc_id": "i1",
                "page": 1,
                "text": "Invoice subtotal EGP 176,700 marked PAID.",
            },
            *self.BANK_RESULTS,
        ]
        gaps = detect_invoice_bank_total_gaps(results, self.DOCS)
        assert gaps, "Bank vs invoice mismatch should be detected"
        g = gaps[0]
        assert abs(g["variance"] - 50000) < 1
        assert g["severity"] == "critical"

    def test_milestone_mismatches_detected(self):
        results = [
            {
                "doc_id": "c1",
                "page": 2,
                "text": (
                    "Schedule: Milestone 1 due EGP 60,000. Milestone 2 due EGP 55,000. "
                    "Milestone 3 due EGP 35,000."
                ),
            },
            {
                "doc_id": "b1",
                "page": 1,
                "text": (
                    "Payment transfer EGP 68,400. Payment transfer EGP 62,930. "
                    "Payment transfer EGP 45,370."
                ),
            },
        ]
        gaps = detect_milestone_payment_mismatch(results, self.DOCS)
        assert gaps, "Milestone mismatches should be detected"
        strict = [g for g in gaps if g["kind"] == "milestone_strict"]
        assert len(strict) >= 2, f"Expected at least 2 strict mismatches, got {len(strict)}"


# ── Entity schema ────────────────────────────────────────────────────────────

class TestEntityAmountRole:
    def test_entity_accepts_amount_role(self):
        e = Entity(
            entity_type="amount",
            value="200,000 EGP",
            normalized_value="200000 EGP",
            source_doc_id="test",
            source_page=1,
            amount_role="total_contract_value",
        )
        assert e.amount_role == "total_contract_value"

    def test_entity_amount_role_defaults_none(self):
        e = Entity(
            entity_type="amount",
            value="100 EGP",
            normalized_value="100 EGP",
            source_doc_id="test",
            source_page=1,
        )
        assert e.amount_role is None

    def test_non_amount_entity_role_is_none(self):
        e = Entity(
            entity_type="company",
            value="Nile Consulting",
            normalized_value="Nile Consulting",
            source_doc_id="test",
            source_page=1,
        )
        assert e.amount_role is None


# ── Planner checklist includes invoice coverage check ────────────────────────

class TestPlannerInvoiceCoverage:
    def test_checklist_includes_invoice_coverage_when_all_three_docs(self):
        raw = generate_checklist.invoke({
            "doc_types_json": json.dumps(["invoice", "contract", "bank_statement"]),
            "doc_ids_json": json.dumps(["a", "b", "c"]),
        })
        data = json.loads(raw)
        descriptions = [c["description"] for c in data["checklist"]]
        assert any(
            "invoice covers all contract periods" in d.lower()
            for d in descriptions
        ), f"Expected invoice coverage check, got: {descriptions}"

    def test_checklist_no_invoice_coverage_without_bank(self):
        raw = generate_checklist.invoke({
            "doc_types_json": json.dumps(["invoice", "contract"]),
            "doc_ids_json": json.dumps(["a", "b"]),
        })
        data = json.loads(raw)
        descriptions = [c["description"] for c in data["checklist"]]
        assert not any(
            "invoice covers all contract periods" in d.lower()
            for d in descriptions
        )
