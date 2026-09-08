"""Planner checklist composition for the document-type combinations we audit."""
import json

from app.tools.planner_tools import generate_checklist


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
