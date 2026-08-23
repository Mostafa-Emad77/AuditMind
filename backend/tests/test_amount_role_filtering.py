"""Tests for amount-role-aware filtering that prevents false-positive contradictions."""
import json

from app.models.schemas import Entity
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

    def test_unknown_vs_anything_is_rejected(self):
        """
        `unknown` previously passed through as comparable-with-anything. Combined with
        untagged running-balance figures, that is what produced conflicts like
        "running balance 750,000 vs line item 25,000". An amount that could not be
        classified is no longer grounds for comparison.
        """
        assert not _roles_are_comparable("unknown", "total_contract_value")
        assert not _roles_are_comparable("unknown", "single_payment")
        assert not _roles_are_comparable("unknown", "unknown")

    def test_milestone_vs_single_payment_is_compatible(self):
        assert _roles_are_comparable("milestone_scheduled", "single_payment")
        assert _roles_are_comparable("single_payment", "milestone_scheduled")

    def test_none_treated_as_unknown_and_rejected(self):
        assert not _roles_are_comparable(None, "total_contract_value")
        assert not _roles_are_comparable(None, None)


# ── False positive regression: opening balance vs invoice ────────────────────

class TestFalsePositiveRegression:
    """These reproduce the exact false-positive scenarios from production."""

    def test_opening_balance_not_compared_to_invoice_total(self):
        """320,000 opening balance must NOT be compared to 180,000 invoice claim."""
        assert not _roles_are_comparable("opening_balance", "total_invoice")

    def test_single_payment_not_compared_to_contract_total(self):
        """50,000 advance payment must NOT be compared to 200,000 contract total."""
        assert not _roles_are_comparable("single_payment", "total_contract_value")


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
            "covered by an invoice" in d.lower() or ("invoice" in d.lower() and "contract" in d.lower())
            for d in descriptions
        ), f"Expected invoice/contract coverage check, got: {descriptions}"

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


# ── Planner checklist: schedule + bank reconciliation coverage ───────────────
# Ported from the deleted test_schedule_gap_detection.py — these two assertions
# don't depend on the removed rule-based detectors, only on generate_checklist.

class TestPlannerScheduleAndBankChecklist:
    def test_generate_checklist_includes_bank_reconciliation_items(self):
        raw = generate_checklist.invoke({
            "doc_types_json": json.dumps(["invoice", "contract", "bank_statement"]),
            "doc_ids_json": json.dumps(["a", "b", "c"]),
        })
        data = json.loads(raw)
        descriptions = [c["description"] for c in data["checklist"]]
        assert any("bank" in d.lower() and "contract" in d.lower() and "total" in d.lower() for d in descriptions), f"No bank-vs-contract check: {descriptions}"
        assert any("bank" in d.lower() and "invoice" in d.lower() and "total" in d.lower() for d in descriptions), f"No bank-vs-invoice check: {descriptions}"
        assert any("milestone" in d.lower() and "bank" in d.lower() for d in descriptions), f"No milestone-vs-bank check: {descriptions}"
