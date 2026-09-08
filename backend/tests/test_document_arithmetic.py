"""Per-document arithmetic self-checks.

The bank statement had its stated Total Debits, which is what exposed the missing
rows. These are the equivalents for the other document types, so a silent extraction
miss is visible outside the bank path too.
"""
from app.models.schemas import DocumentMeta
from app.services.reconciliation_payload import _document_integrity_warnings

INVOICE = DocumentMeta(doc_id="inv", filename="invoice.pdf", doc_type="invoice")
SHEET = DocumentMeta(doc_id="bs", filename="balance_sheet.pdf", doc_type="balance_sheet")
CONTRACT = DocumentMeta(doc_id="ctr", filename="contract.pdf", doc_type="contract")


def _row(doc_id: str, role: str, value: str) -> dict:
    return {"doc_id": doc_id, "amount_role": role, "normalized_value": value, "value": value}


class TestInvoiceArithmetic:
    def test_subtotal_plus_vat_matching_total_is_silent(self):
        rows = [_row("inv", "invoice_subtotal", "155000 EGP"),
                _row("inv", "vat_tax", "21700 EGP"),
                _row("inv", "total_invoice", "176700 EGP")]
        assert _document_integrity_warnings(rows, [INVOICE]) == []

    def test_mismatch_is_reported(self):
        rows = [_row("inv", "invoice_subtotal", "155000 EGP"),
                _row("inv", "vat_tax", "21700 EGP"),
                _row("inv", "total_invoice", "200000 EGP")]
        warnings = _document_integrity_warnings(rows, [INVOICE])
        assert len(warnings) == 1
        assert "invoice.pdf" in warnings[0] and "176,700.00" in warnings[0]

    def test_incomplete_triple_is_not_guessed_at(self):
        rows = [_row("inv", "invoice_subtotal", "155000 EGP"),
                _row("inv", "total_invoice", "200000 EGP")]
        assert _document_integrity_warnings(rows, [INVOICE]) == []


class TestBalanceSheetArithmetic:
    def test_balanced_sheet_is_silent(self):
        rows = [_row("bs", "total_assets", "2480000 EGP"),
                _row("bs", "total_liabilities", "1510000 EGP"),
                _row("bs", "total_equity", "970000 EGP")]
        assert _document_integrity_warnings(rows, [SHEET]) == []

    def test_unbalanced_sheet_is_reported(self):
        rows = [_row("bs", "total_assets", "2480000 EGP"),
                _row("bs", "total_liabilities", "1510000 EGP"),
                _row("bs", "total_equity", "800000 EGP")]
        warnings = _document_integrity_warnings(rows, [SHEET])
        assert len(warnings) == 1
        assert "total assets" in warnings[0]


class TestContractSchedule:
    def test_schedule_summing_to_the_total_is_silent(self):
        rows = [_row("ctr", "total_contract_value", "200000 EGP"),
                _row("ctr", "milestone_scheduled", "50000 EGP"),
                _row("ctr", "milestone_scheduled", "60000 EGP"),
                _row("ctr", "milestone_scheduled", "90000 EGP")]
        assert _document_integrity_warnings(rows, [CONTRACT]) == []

    def test_shortfall_suggests_missing_schedule_lines(self):
        rows = [_row("ctr", "total_contract_value", "200000 EGP"),
                _row("ctr", "milestone_scheduled", "50000 EGP"),
                _row("ctr", "milestone_scheduled", "60000 EGP")]
        warnings = _document_integrity_warnings(rows, [CONTRACT])
        assert len(warnings) == 1
        assert "may be missing" in warnings[0]

    def test_a_single_milestone_is_not_a_schedule(self):
        """One advance payment is not evidence of a missing schedule."""
        rows = [_row("ctr", "total_contract_value", "200000 EGP"),
                _row("ctr", "milestone_scheduled", "50000 EGP")]
        assert _document_integrity_warnings(rows, [CONTRACT]) == []

    def test_overshoot_is_left_to_the_contradiction_checks(self):
        rows = [_row("ctr", "total_contract_value", "200000 EGP"),
                _row("ctr", "milestone_scheduled", "150000 EGP"),
                _row("ctr", "milestone_scheduled", "150000 EGP")]
        assert _document_integrity_warnings(rows, [CONTRACT]) == []


def test_documents_without_the_needed_roles_are_silent():
    assert _document_integrity_warnings([], [INVOICE, SHEET, CONTRACT]) == []
