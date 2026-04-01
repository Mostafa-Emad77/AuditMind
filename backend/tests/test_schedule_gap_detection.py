import json

from app.agents.cross_checker import (
    detect_bank_overpayment_gaps,
    detect_invoice_bank_total_gaps,
    detect_milestone_payment_mismatch,
    detect_schedule_gaps,
    detect_subtotal_component_gaps,
)
from app.models.schemas import DocumentMeta
from app.tools.planner_tools import generate_checklist


def test_detect_missing_march_retainer_gap():
    docs = [
        DocumentMeta(doc_id="contract-doc", filename="contract_CTR2024044.pdf", doc_type="contract"),
        DocumentMeta(doc_id="invoice-doc", filename="invoice_001_Q1.pdf", doc_type="invoice"),
    ]
    results = [
        {
            "doc_id": "contract-doc",
            "page": 2,
            "text": (
                "Payment schedule: January retainer EGP 50,000, "
                "February retainer EGP 50,000, March retainer EGP 25,000."
            ),
        },
        {
            "doc_id": "invoice-doc",
            "page": 1,
            "text": (
                "Invoice subtotal EGP 155,000 covers January retainer EGP 50,000 "
                "and February retainer EGP 50,000 plus DD report and Risk."
            ),
        },
    ]

    gaps = detect_schedule_gaps(results, docs)
    assert gaps, "Expected at least one schedule gap"
    march = next((g for g in gaps if g.get("month") == "March"), None)
    assert march is not None, f"Expected March gap, got: {gaps}"
    assert march["missing_amount"] >= 25000


def test_generate_checklist_includes_schedule_consistency_item():
    raw = generate_checklist.invoke({
        "doc_types_json": json.dumps(["invoice", "contract"]),
        "doc_ids_json": json.dumps(["a", "b"]),
    })
    data = json.loads(raw)
    descriptions = [c["description"] for c in data["checklist"]]
    assert any("payment schedule milestones" in d for d in descriptions)
    assert any("invoice subtotal against explicitly listed line items/components" in d for d in descriptions)


def test_detect_invoice_subtotal_component_gap():
    docs = [
        DocumentMeta(doc_id="invoice-doc", filename="invoice_001_Q1.pdf", doc_type="invoice"),
    ]
    results = [
        {
            "doc_id": "invoice-doc",
            "page": 1,
            "text": (
                "Invoice subtotal EGP 155,000 includes only January retainer EGP 50,000 "
                "and February retainer EGP 50,000."
            ),
        },
    ]
    gaps = detect_subtotal_component_gaps(results, docs)
    assert gaps, "Expected subtotal/component reconciliation gap"
    g = gaps[0]
    assert g["difference"] >= 50000
    assert g["severity"] in ("critical", "warning")


def test_detect_bank_overpayment_vs_contract_total():
    """Bank 226,700 vs contract 200,000 => variance beyond tolerance (plan scenario)."""
    docs = [
        DocumentMeta(doc_id="c1", filename="contract.pdf", doc_type="contract"),
        DocumentMeta(doc_id="b1", filename="bank_statement.pdf", doc_type="bank_statement"),
    ]
    results = [
        {
            "doc_id": "c1",
            "page": 1,
            "text": "Article 3 contract value EGP 200,000 for the full engagement.",
        },
        {
            "doc_id": "b1",
            "page": 1,
            "text": (
                "Payment transfer EGP 50,000. Payment transfer EGP 68,400. "
                "Payment transfer EGP 62,930. Payment transfer EGP 45,370."
            ),
        },
    ]
    gaps = detect_bank_overpayment_gaps(results, docs)
    assert gaps, "Expected bank vs contract gap"
    g = gaps[0]
    assert abs(g["variance"] - 26700) < 1
    assert g["severity"] == "critical"


def test_detect_invoice_total_vs_bank_total():
    """Invoice 176,700 vs bank 226,700 => large variance (plan scenario)."""
    docs = [
        DocumentMeta(doc_id="i1", filename="invoice.pdf", doc_type="invoice"),
        DocumentMeta(doc_id="b1", filename="bank_statement.pdf", doc_type="bank_statement"),
    ]
    results = [
        {
            "doc_id": "i1",
            "page": 1,
            "text": "Invoice subtotal EGP 176,700 marked PAID.",
        },
        {
            "doc_id": "b1",
            "page": 1,
            "text": (
                "Payment transfer EGP 50,000. Payment transfer EGP 68,400. "
                "Payment transfer EGP 62,930. Payment transfer EGP 45,370."
            ),
        },
    ]
    gaps = detect_invoice_bank_total_gaps(results, docs)
    assert gaps, "Expected invoice vs bank gap"
    g = gaps[0]
    assert abs(g["variance"] - 50000) < 1
    assert g["severity"] == "critical"


def test_detect_milestone_strict_and_cumulative_mismatch():
    """Contract milestones 60k/55k/35k vs bank payments 50k/50k/50k — strict + cumulative signals."""
    docs = [
        DocumentMeta(doc_id="c1", filename="contract.pdf", doc_type="contract"),
        DocumentMeta(doc_id="b1", filename="bank_statement.pdf", doc_type="bank_statement"),
    ]
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
                "Payment transfer EGP 50,000 advance. Payment transfer EGP 50,000. "
                "Payment transfer EGP 50,000."
            ),
        },
    ]
    gaps = detect_milestone_payment_mismatch(results, docs)
    assert gaps, "Expected milestone vs bank mismatches"
    kinds = {g.get("kind") for g in gaps}
    assert "milestone_strict" in kinds
    assert "milestone_cumulative" in kinds
    assert any(g.get("severity") == "critical" for g in gaps if g.get("kind") == "milestone_cumulative")


def test_generate_checklist_includes_bank_reconciliation_items():
    raw = generate_checklist.invoke({
        "doc_types_json": json.dumps(["invoice", "contract", "bank_statement"]),
        "doc_ids_json": json.dumps(["a", "b", "c"]),
    })
    data = json.loads(raw)
    descriptions = [c["description"] for c in data["checklist"]]
    assert any("total bank payments against contract total value" in d.lower() for d in descriptions)
    assert any("sum of bank payments against invoice total" in d.lower() for d in descriptions)
    assert any("milestone payment schedule" in d.lower() and "actual bank payments" in d.lower() for d in descriptions)
