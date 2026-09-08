"""Title-first classification, invoice-total picking, table-row chunking."""
from app.services.document_processor import (
    _chunk_table_rows,
    _classify_by_title,
    _classify_document_type,
    _looks_like_header,
)
from app.services.reconciliation_payload import _pick_invoice_total


class TestTitleClassifier:
    # (title block, filename, expected) — the six types that collided in the run where
    # a certificate, a board resolution and a QS report were typed invoice/invoice/contract.
    CASES = [
        ("", "doc4_payment_cert_PC-2025-007.pdf", "payment_certificate"),
        ("شهادة استحقاق الدفع\nPC-2025-007\nCONTRACT NO. BLD-2024-019\nAmount (EGP) VAT Total",
         "scan_0001.pdf", "payment_certificate"),
        ("Board of Directors Meeting Minutes & Resolution\nBR-2025-003\nVAT total amount",
         "x.pdf", "board_resolution"),
        ("", "doc6_board_resolution_BR-2025-003.pdf", "board_resolution"),
        ("Al-Daqqa Quantity Surveying & Engineering\nQS-RPT-025\nMilestone 3 Certification Report",
         "x.pdf", "qs_report"),
        # An invoice cites the contract it bills against — it must not become a contract.
        ("Pyramid Steel\nINV-BLD-2025-031\nagainst Contract BLD-2024-019\nsubtotal VAT total due",
         "doc3_invoice_INV-BLD-2025-031.pdf", "invoice"),
        ("Subcontract Agreement — Structural Works\nContract No. BLD-2024-019\nbetween A and B",
         "doc1_contract_BLD2024019.pdf", "contract"),
        ("lorem ipsum dolor", "scan.pdf", None),
    ]

    def test_title_block_and_filename_decide_the_type(self):
        for head, filename, expected in self.CASES:
            assert _classify_by_title(head, filename) == expected, filename

    def test_title_beats_vocabulary_scoring(self):
        """Invoice vocabulary must not override a certificate's title."""
        text = (
            "شهادة استحقاق الدفع\nPC-2025-007\n" + "invoice total amount due VAT subtotal " * 20
        )
        assert _classify_document_type(text, "cert.pdf") == "payment_certificate"


class TestInvoiceTotalPicker:
    def test_prefers_arithmetically_supported_total(self):
        # subtotal + VAT == 36,465,379 ; a larger stray "total" must lose
        by_doc = {"inv": {"total": [36465379.0, 38419938.0], "subtotal": [31993550.0], "vat": [4471829.0]}}
        assert _pick_invoice_total(by_doc) == 36465379.0

    def test_falls_back_to_max_total_when_no_arithmetic_support(self):
        by_doc = {"inv": {"total": [100.0, 250.0], "subtotal": [], "vat": []}}
        assert _pick_invoice_total(by_doc) == 250.0

    def test_falls_back_to_subtotal(self):
        by_doc = {"inv": {"total": [], "subtotal": [900.0], "vat": [126.0]}}
        assert _pick_invoice_total(by_doc) == 900.0

    def test_empty(self):
        assert _pick_invoice_total({}) is None


class TestTableRowChunking:
    def test_header_detection(self):
        assert _looks_like_header(["Date", "Ref No.", "Description", "Debit (EGP)", "Credit", "Balance"])
        assert not _looks_like_header(["05/01/2025", "TXN-002", "Payment", "50,000.00", "", "750,000.00"])

    def test_rows_never_split_and_header_repeats(self):
        header = "Date | Ref | Description | Debit | Credit | Balance"
        rows = [f"0{i}/01/2025 | TXN-00{i} | Payment to vendor number {i} long description | {i}0,000.00 | | 1,000.00"
                for i in range(1, 9)]
        chunks = _chunk_table_rows(header, rows, limit=260)
        assert len(chunks) > 1
        for ch in chunks:
            lines = ch.split("\n")
            assert lines[0].startswith("[TABLE ROW] columns: " + header)
            for line in lines[1:]:
                assert line.startswith("[TABLE ROW] ")
                assert line.count("|") == 5  # a full row, never a fragment
        # every row present exactly once
        all_lines = [l for ch in chunks for l in ch.split("\n")[1:]]
        assert len(all_lines) == len(rows)
