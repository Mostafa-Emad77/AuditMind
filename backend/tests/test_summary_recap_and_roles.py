"""A statement's own recap is not a transaction, and a bank statement has no schedule.

From the run where page 1 reconciled exactly (10 debits = 1,185,700, the stated Total
Debits) but page 2's recap block —

    TXN-002: EGP 50,000 (advance)   ...   Bank total paid: EGP 226,700.00

— was extracted as transactions too, pushing the total to 1,412,400, and its
"(milestone 1)" wording tagged bank amounts `milestone_scheduled`, which paired every
contract schedule line with every bank payment.
"""
import app.services.reconciliation_payload as rp
from app.models.schemas import DocumentMeta
from app.services.entity_extractor import _role_for_doc_type

BANK = DocumentMeta(doc_id="bank1", filename="bank.pdf", doc_type="bank_statement")
CONTRACT = DocumentMeta(doc_id="c1", filename="contract.pdf", doc_type="contract")


def _debit(value: str, *, ref=None, date=None, page=1, anchors=None) -> dict:
    return {
        "doc_id": "bank1", "amount_role": "transaction_debit",
        "normalized_value": value, "value": value, "source_page": page,
        "txn_ref": ref, "txn_date": date, "anchor_ids": anchors or [],
    }


def _stated(value: str) -> dict:
    return {"doc_id": "bank1", "amount_role": "statement_total_debits",
            "normalized_value": value, "value": value, "source_page": 1}


class TestRecapIsNotATransaction:
    def test_identified_rows_win_over_an_unidentified_recap(self):
        """The exact regression: rows total 1,185,700 plus a 226,700 recap = 1,412,400."""
        rows = [
            _debit("1000000 EGP", ref="TXN-001", date="2025-01-05"),
            _debit("185700 EGP", ref="TXN-002", date="2025-02-05"),
            _debit("1185700 EGP"),                      # "Bank total paid" recap, no identity
            _stated("1185700 EGP"),
        ]
        snap = rp._aggregate_snapshot(rows, [BANK])
        assert snap.bank_paid_total == 1_185_700.0
        assert snap.bank_extraction_incomplete is False

    def test_same_ref_recapped_on_a_later_page_is_one_transaction(self):
        rows = [
            _debit("50000 EGP", ref="TXN-002", date="2025-01-05", page=1),
            _debit("50000 EGP", ref="TXN-002", page=2),   # recap of the same row
            _stated("50000 EGP"),
        ]
        snap = rp._aggregate_snapshot(rows, [BANK])
        assert snap.bank_paid_total == 50_000.0

    def test_statements_with_no_identified_rows_still_count(self):
        """Fallback: when nothing carries a ref or date, every debit is all we have."""
        rows = [_debit("30000 EGP", page=1), _debit("20000 EGP", page=2), _stated("50000 EGP")]
        snap = rp._aggregate_snapshot(rows, [BANK])
        assert snap.bank_paid_total == 50_000.0

    def test_distinct_refs_with_equal_amounts_stay_distinct(self):
        rows = [
            _debit("25000 EGP", ref="TXN-001", date="2025-01-01"),
            _debit("25000 EGP", ref="TXN-002", date="2025-02-01"),
            _stated("50000 EGP"),
        ]
        snap = rp._aggregate_snapshot(rows, [BANK])
        assert snap.bank_paid_total == 50_000.0


class TestDocTypeRoleGuard:
    def test_bank_statement_cannot_hold_a_contract_schedule(self):
        assert _role_for_doc_type("milestone_scheduled", "bank_statement") == "unknown"
        assert _role_for_doc_type("total_contract_value", "bank_statement") == "unknown"
        assert _role_for_doc_type("total_invoice", "bank_statement") == "unknown"

    def test_bank_roles_pass_through(self):
        for role in ("transaction_debit", "transaction_credit", "running_balance",
                     "statement_total_debits", "opening_balance", "unknown"):
            assert _role_for_doc_type(role, "bank_statement") == role

    def test_other_doc_types_are_unconstrained(self):
        assert _role_for_doc_type("milestone_scheduled", "contract") == "milestone_scheduled"
        assert _role_for_doc_type("total_invoice", "invoice") == "total_invoice"
        assert _role_for_doc_type("milestone_scheduled", "unknown") == "milestone_scheduled"


class TestInvoiceArithmeticIncludesLateFee:
    def _rows(self, total, subtotal, vat, fee=None):
        out = [
            {"doc_id": "inv", "amount_role": "total_invoice", "normalized_value": total, "value": total},
            {"doc_id": "inv", "amount_role": "invoice_subtotal", "normalized_value": subtotal, "value": subtotal},
            {"doc_id": "inv", "amount_role": "vat_tax", "normalized_value": vat, "value": vat},
        ]
        if fee:
            out.append({"doc_id": "inv", "amount_role": "late_fee",
                        "normalized_value": fee, "value": fee})
        return out

    INVOICE = DocumentMeta(doc_id="inv", filename="invoice_002_Q2.pdf", doc_type="invoice")

    def test_late_fee_completes_the_total(self):
        """150,000 + 21,000 + 3,534 = 174,534 — the warning was a false positive."""
        rows = self._rows("174534 EGP", "150000 EGP", "21000 EGP", "3534 EGP")
        assert rp._document_integrity_warnings(rows, [self.INVOICE]) == []

    def test_a_genuine_mismatch_still_reports_and_shows_the_fee(self):
        rows = self._rows("180000 EGP", "150000 EGP", "21000 EGP", "3534 EGP")
        warnings = rp._document_integrity_warnings(rows, [self.INVOICE])
        assert len(warnings) == 1
        assert "late fee" in warnings[0]

    def test_no_fee_keeps_the_plain_identity(self):
        rows = self._rows("171000 EGP", "150000 EGP", "21000 EGP")
        assert rp._document_integrity_warnings(rows, [self.INVOICE]) == []


# ── Attributing a payment to the contract it names ───────────────────────────

class TestPaymentAttribution:
    """
    Attribution relied on an LLM-emitted RELATES edge, which existed for only one of
    four payments to the contract's counterparty — so Bank Paid Total read 45,370
    instead of 226,700. Rows survive ingestion intact now, so a payment can be tied
    to the contract named in its own description.
    """

    ROWS = [
        "[TABLE ROW] 05/01/2025 | TXN-002 | Payment to Nile - Advance عقد CTR-2024-044 | 50,000.00",
        "[TABLE ROW] 12/01/2025 | TXN-003 | Office Rent Q1 2025 | 75,000.00",
        "[TABLE ROW] 31/01/2025 | TXN-005 | Payment to Nile - Invoice INV-2025-001 | 68,400.00",
        "[TABLE ROW] 14/02/2025 | TXN-007 | Staff Salaries February | 210,000.00",
    ]

    def _linked(self, monkeypatch, cores, refs=("TXN-002", "TXN-003", "TXN-005", "TXN-007")):
        monkeypatch.setattr(rp, "get_all_chunks_for_docs",
                            lambda _ids: [{"text": "\n".join(self.ROWS)}])
        return rp._refs_named_alongside_an_audit_identifier(["bank1"], set(refs), set(cores))

    def test_only_rows_naming_an_audited_identifier_are_linked(self, monkeypatch):
        linked = self._linked(monkeypatch, {"ctr2024044", "inv2025001"})
        assert linked == {"TXN-002", "TXN-005"}   # rent and salaries excluded

    def test_punctuation_in_the_identifier_does_not_matter(self, monkeypatch):
        """"CTR-2024-044" in the row vs "ctr2024044" as the core."""
        assert "TXN-002" in self._linked(monkeypatch, {"ctr2024044"})

    def test_no_identifiers_links_nothing(self, monkeypatch):
        assert self._linked(monkeypatch, set()) == set()

    def test_unreadable_rows_degrade_quietly(self, monkeypatch):
        def _boom(_ids):
            raise RuntimeError("qdrant down")
        monkeypatch.setattr(rp, "get_all_chunks_for_docs", _boom)
        assert rp._refs_named_alongside_an_audit_identifier(
            ["bank1"], {"TXN-002"}, {"ctr2024044"}) == set()

    def test_text_attribution_counts_without_a_graph_edge(self):
        """The regression: only TXN-012 had an edge, so only it was counted."""
        rows = [
            dict(_debit("50000 EGP", ref="TXN-002", date="2025-01-05"), anchor_ids=[]),
            dict(_debit("68400 EGP", ref="TXN-005", date="2025-01-31"), anchor_ids=[]),
            dict(_debit("45370 EGP", ref="TXN-012", date="2025-03-25"), anchor_ids=["ctr"]),
            dict(_debit("210000 EGP", ref="TXN-007", date="2025-02-14"), anchor_ids=[]),
            _stated("373770 EGP"),
        ]
        snap = rp._aggregate_snapshot(
            rows, [BANK, CONTRACT], {"ctr"}, {"TXN-002", "TXN-005", "TXN-012"},
        )
        assert snap.bank_paid_total == 163_770.0     # salaries correctly excluded
        assert snap.bank_extraction_incomplete is False

    def test_ref_matching_is_case_insensitive(self):
        """Debit refs are stored lowercased; the linked set keeps source casing."""
        rows = [dict(_debit("50000 EGP", ref="TXN-002", date="2025-01-05"), anchor_ids=[]),
                _stated("50000 EGP")]
        snap = rp._aggregate_snapshot(rows, [BANK, CONTRACT], {"ctr"}, {"TXN-002"})
        assert snap.bank_paid_total == 50_000.0


# ── Currency follows the documents, not a hardcoded default ──────────────────

class TestSnapshotCurrency:
    """Every card was labelled EGP regardless of the documents audited."""

    def _rows(self, *codes):
        return [
            dict(_debit(f"{(i + 1) * 1000} EGP", ref=f"TXN-00{i}", date=f"2025-01-0{i + 1}"),
                 amount_currency=code)
            for i, code in enumerate(codes)
        ]

    def test_currency_comes_from_the_amounts(self):
        snap = rp._aggregate_snapshot(self._rows("USD", "USD"), [BANK])
        assert snap.currency == "USD"

    def test_majority_wins_and_the_mix_is_flagged(self):
        snap = rp._aggregate_snapshot(self._rows("SAR", "SAR", "USD"), [BANK])
        assert snap.currency == "SAR"
        assert any("more than one currency" in w for w in snap.integrity_warnings)

    def test_single_currency_raises_no_warning(self):
        snap = rp._aggregate_snapshot(self._rows("AED", "AED"), [BANK])
        assert snap.integrity_warnings == []

    def test_falls_back_when_nothing_is_tagged(self):
        snap = rp._aggregate_snapshot(self._rows(None, None), [BANK])
        assert snap.currency == "EGP"
