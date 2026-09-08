"""Regression tests: bank-total inflation, role comparability, identifier
normalisation, signatory fields."""
from typing import get_args

import app.services.reconciliation_payload as rp
from app.models.schemas import (
    AMOUNT_ROLES,
    BANK_OUTFLOW_ROLES,
    NON_SUMMABLE_ROLES,
    DocumentMeta,
    Entity,
)
from app.services.graph_builder import _role_pair_reason, _roles_are_comparable
from app.utils.canonical_id import canonical_entity_id, normalize_identifier


def _bank_row(role: str, value: str, page: int = 1, doc_id: str = "bank1") -> dict:
    return {
        "doc_id": doc_id,
        "amount_role": role,
        "normalized_value": value,
        "value": value,
        "source_page": page,
        "entity_id": f"{role}-{value}-{page}",
    }


BANK_DOC = DocumentMeta(doc_id="bank1", filename="bank.pdf", doc_type="bank_statement")


# ── BUG 1: running balances must never inflate the bank total ────────────────

class TestBankTotalExcludesBalances:
    def test_running_balances_are_not_summed(self):
        """The original bug: balance-column figures dwarfed the actual debits."""
        rows = [
            _bank_row("transaction_debit", "50000 EGP", page=1),
            _bank_row("transaction_debit", "62930 EGP", page=2),
            # Balance column values — huge, cumulative, and previously summed in.
            _bank_row("running_balance", "800000 EGP", page=1),
            _bank_row("running_balance", "750000 EGP", page=2),
            _bank_row("opening_balance", "320000 EGP", page=1),
            _bank_row("closing_balance", "374300 EGP", page=3),
        ]
        snap = rp._aggregate_snapshot(rows, [BANK_DOC])
        assert snap.bank_paid_total == 112930.0

    def test_unknown_role_amounts_are_not_summed(self):
        """`unknown` used to be summed when >= 100 — that is what swept balances in."""
        rows = [
            _bank_row("transaction_debit", "50000 EGP", page=1),
            _bank_row("unknown", "999999 EGP", page=2),
        ]
        snap = rp._aggregate_snapshot(rows, [BANK_DOC])
        assert snap.bank_paid_total == 50000.0

    def test_credits_are_not_counted_as_money_paid(self):
        rows = [
            _bank_row("transaction_debit", "50000 EGP", page=1),
            _bank_row("transaction_credit", "480000 EGP", page=1),
        ]
        snap = rp._aggregate_snapshot(rows, [BANK_DOC])
        assert snap.bank_paid_total == 50000.0

    def test_statement_total_is_used_to_validate_not_to_sum(self):
        """The stated Total Debits must validate the rows, never be added to them."""
        rows = [
            _bank_row("transaction_debit", "100000 EGP", page=1),
            _bank_row("transaction_debit", "126700 EGP", page=2),
            _bank_row("statement_total_debits", "226700 EGP", page=1),
        ]
        snap = rp._aggregate_snapshot(rows, [BANK_DOC])
        assert snap.bank_paid_total == 226700.0
        assert snap.notes is None or "unverified" not in (snap.notes or "")

    def test_validation_withholds_mismatched_total(self):
        """
        When extracted debits don't reconcile with the statement's stated Total
        Debits, the bank total is WITHHELD (not rendered as a wrong number with a
        caption beside it) and an explicit incomplete state is surfaced instead.
        """
        rows = [
            _bank_row("transaction_debit", "100000 EGP", page=1),
            _bank_row("statement_total_debits", "226700 EGP", page=1),
        ]
        snap = rp._aggregate_snapshot(rows, [BANK_DOC])
        assert snap.bank_paid_total is None
        assert snap.variance_vs_contract is None
        assert snap.bank_extraction_incomplete is True
        assert snap.bank_debits_verified == 100000.0
        assert snap.bank_debits_stated == 226700.0
        assert "extraction incomplete" in (snap.notes or "").lower()


class TestBankPaymentAttribution:
    """
    "Bank paid" must mean money paid against THIS contract, not every outflow on the
    statement — otherwise salaries and utility bills land in a figure that gets compared
    against a contract total.
    """

    CONTRACT_DOC = DocumentMeta(doc_id="c1", filename="contract.pdf", doc_type="contract")

    def test_only_contract_linked_debits_are_counted(self):
        rows = [
            dict(_bank_row("transaction_debit", "50000 EGP", page=1), anchor_ids=["vendor-x"]),
            dict(_bank_row("transaction_debit", "176700 EGP", page=2), anchor_ids=["vendor-x"]),
            # Unrelated outflows — salaries, utilities, other vendors.
            dict(_bank_row("transaction_debit", "210000 EGP", page=2), anchor_ids=["payroll"]),
            dict(_bank_row("transaction_debit", "18000 EGP", page=2), anchor_ids=[]),
        ]
        snap = rp._aggregate_snapshot(rows, [BANK_DOC, self.CONTRACT_DOC], {"vendor-x"})
        assert snap.bank_paid_total == 226700.0

    def test_falls_back_to_all_debits_without_a_contract(self):
        """A bank-statement-only audit has nothing to scope to, so report all outflows."""
        rows = [
            dict(_bank_row("transaction_debit", "50000 EGP", page=1), anchor_ids=[]),
            dict(_bank_row("transaction_debit", "18000 EGP", page=2), anchor_ids=[]),
        ]
        snap = rp._aggregate_snapshot(rows, [BANK_DOC], set())
        assert snap.bank_paid_total == 68000.0

    def test_statement_validation_uses_all_debits_not_the_scoped_subset(self):
        """The extraction check covers every row; contract scoping must not skew it."""
        rows = [
            dict(_bank_row("transaction_debit", "226700 EGP", page=1), anchor_ids=["vendor-x"]),
            dict(_bank_row("transaction_debit", "773300 EGP", page=2), anchor_ids=["payroll"]),
            dict(_bank_row("statement_total_debits", "1000000 EGP", page=1), anchor_ids=[]),
        ]
        snap = rp._aggregate_snapshot(rows, [BANK_DOC, self.CONTRACT_DOC], {"vendor-x"})
        assert snap.bank_paid_total == 226700.0
        # All debits (226,700 + 773,300) match the stated 1,000,000 → no warning.
        assert "extraction incomplete" not in (snap.notes or "").lower()


class TestRoleConstants:
    def test_non_summable_roles_are_all_real_roles(self):
        valid = set(get_args(AMOUNT_ROLES))
        assert NON_SUMMABLE_ROLES <= valid
        assert BANK_OUTFLOW_ROLES <= valid

    def test_outflow_and_non_summable_are_disjoint(self):
        assert not (NON_SUMMABLE_ROLES & BANK_OUTFLOW_ROLES)


# ── BUG 2: only logically comparable roles may be paired ─────────────────────

class TestRoleComparability:
    def test_running_balance_is_never_comparable(self):
        """The exact nonsense pairing from the bug report."""
        assert not _roles_are_comparable("running_balance", "invoice_line_item")
        assert not _roles_are_comparable("running_balance", "total_contract_value")
        assert not _roles_are_comparable("running_balance", "running_balance")

    def test_unknown_roles_no_longer_pass_through(self):
        """`unknown` used to be treated as comparable with anything."""
        assert not _roles_are_comparable("unknown", "total_contract_value")
        assert not _roles_are_comparable("total_invoice", "unknown")
        assert not _roles_are_comparable("unknown", "unknown")

    def test_statement_totals_are_never_comparable(self):
        assert not _roles_are_comparable("statement_total_debits", "total_invoice")

    def test_contract_total_vs_invoice_restatement_is_comparable(self):
        """The restatement check must survive — it is a real finding."""
        assert _roles_are_comparable(
            "total_contract_value", "invoice_referenced_contract_value"
        )

    def test_contract_total_vs_invoice_total_is_comparable(self):
        assert _roles_are_comparable("total_contract_value", "total_invoice")

    def test_milestone_vs_bank_payment_is_comparable(self):
        assert _roles_are_comparable("milestone_scheduled", "transaction_debit")

    def test_retainer_vs_retainer_is_comparable(self):
        """Needed to catch an unauthorized retainer rate increase."""
        assert _roles_are_comparable("retainer", "retainer")

    def test_incomparable_pair_stays_rejected(self):
        assert not _roles_are_comparable("retainer", "total_contract_value")

    def test_comparability_is_symmetric(self):
        roles = list(get_args(AMOUNT_ROLES))
        for a in roles:
            for b in roles:
                assert _roles_are_comparable(a, b) == _roles_are_comparable(b, a), (a, b)

    def test_reason_is_human_readable(self):
        reason = _role_pair_reason("total_contract_value", "invoice_referenced_contract_value")
        assert reason is not None
        assert "contract" in reason.lower()
        # Not an opaque machine token.
        assert "_" not in reason


# ── BUG 3: identifier variants must resolve to one node ──────────────────────

class TestIdentifierResolution:
    def test_formatting_variants_collapse(self):
        variants = ["CTR-2024-044", "ctr 2024 044", "CTR2024044", "CTR/2024/044"]
        ids = {canonical_entity_id("contract_id", v, doc_id="doc-a") for v in variants}
        assert len(ids) == 1, f"variants produced {len(ids)} distinct nodes"

    def test_same_identifier_across_documents_is_one_node(self):
        a = canonical_entity_id("contract_id", "CTR-2024-044", doc_id="contract-doc")
        b = canonical_entity_id("contract_id", "CTR2024044", doc_id="invoice-doc")
        assert a == b

    def test_genuinely_different_identifiers_stay_distinct(self):
        a = canonical_entity_id("contract_id", "CTR-2024-044", doc_id="d")
        b = canonical_entity_id("contract_id", "CTR-2024-045", doc_id="d")
        assert a != b

    def test_company_names_are_not_punctuation_stripped(self):
        """Legal-suffix punctuation is meaningful for companies — don't over-merge."""
        a = canonical_entity_id("company", "Delta Trading", doc_id="d")
        b = canonical_entity_id("company", "Delta Trading S.A.E.", doc_id="d")
        assert a != b

    def test_normalize_identifier_helper(self):
        assert normalize_identifier("CTR-2024-044") == "ctr2024044"
        assert normalize_identifier("  inv 001  ") == "inv001"


# ── BUG 4: Person / signatory entities ───────────────────────────────────────

class TestPersonEntity:
    def test_person_carries_authority_fields(self):
        e = Entity(
            entity_type="person",
            value="Karim Fawzy",
            normalized_value="Karim Fawzy",
            source_doc_id="d1",
            source_page=1,
            role_title="Chairman",
            signing_authority_level="chairman",
            document_signed="contract CTR-2024-044",
        )
        assert e.signing_authority_level == "chairman"
        assert e.document_signed == "contract CTR-2024-044"

    def test_person_fields_default_to_none(self):
        e = Entity(
            entity_type="amount",
            value="100 EGP",
            normalized_value="100 EGP",
            source_doc_id="d1",
            source_page=1,
        )
        assert e.role_title is None
        assert e.signing_authority_level is None

    def test_same_person_dedupes_across_documents(self):
        """Required so one signatory accumulates everything they signed."""
        a = canonical_entity_id("person", "Karim Fawzy", doc_id="contract")
        b = canonical_entity_id("person", "karim fawzy", doc_id="amendment")
        assert a == b


# ── Document classification regression ───────────────────────────────────────

class TestHeuristicClassification:
    """
    A contract that references invoices and VAT was scoring as an `invoice`, which
    corrupted reconciliation (the snapshot branches on doc_type, so the contract's
    total was never picked up as contract_total).
    """

    def _classify(self, text: str, filename: str) -> str:
        from app.services.document_processor import _classify_document_type_heuristic
        return _classify_document_type_heuristic(text, filename)

    def test_contract_mentioning_invoices_is_still_a_contract(self):
        text = (
            "CONTRACT AGREEMENT between Party A and Party B. "
            "This contract is governed by Egyptian law. Contract total inclusive of VAT. "
            "Termination clause. The parties agree. Contract CTR-2024-044. "
            "Subtotal and invoice INV-2025-001 referenced herein. "
            "Contract effective date. Contract value. Contract schedule. Contract penalties."
        )
        assert self._classify(text, "contract_CTR2024044.pdf") == "contract"

    def test_filename_hint_survives_underscore_separators(self):
        """`\bcontract\b` cannot match `contract_X.pdf` — separators must be normalized."""
        neutral = "This document concerns an agreement between the parties."
        assert self._classify(neutral, "contract_CTR2024044.pdf") == "contract"

    def test_repeated_terms_outweigh_a_passing_mention(self):
        text = "invoice. " + ("This contract and agreement. " * 6)
        assert self._classify(text, "doc.pdf") == "contract"

    def test_real_invoice_still_classifies_as_invoice(self):
        text = (
            "TAX INVOICE. Invoice No: INV-2025-001. Bill To: Delta Trading. "
            "Subtotal 155,000. VAT 14%. Total Amount Due 176,700. Due date 2025-04-15."
        )
        assert self._classify(text, "invoice_001_Q1.pdf") == "invoice"

    def test_bank_statement_still_classifies(self):
        text = (
            "BANK STATEMENT. Account Statement Q1 2025. IBAN EG38. "
            "Transaction register. Debit Credit Balance. Opening balance. Closing balance."
        )
        assert self._classify(text, "bank_statement_Q1_2025.pdf") == "bank_statement"


# ── Role filter must be applied before LIMIT ─────────────────────────────────

class TestRoleFilterAppliedInQuery:
    """Role allow-list must run inside the query, before ranking/limit."""

    def _captured_cypher(self, monkeypatch) -> str:
        import app.services.graph_builder as gb
        captured = {}

        class _FakeSession:
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def run(self, query, **kwargs):
                captured["query"] = query
                captured["params"] = kwargs
                return []

        class _FakeDriver:
            def session(self, **kwargs): return _FakeSession()

        monkeypatch.setattr(gb, "_run_with_reconnect", lambda fn: fn(_FakeDriver()))
        gb.find_contradictions(["d1", "d2"])
        return captured["query"], captured["params"]

    def test_allow_list_is_filtered_in_cypher_before_limit(self, monkeypatch):
        query, _params = self._captured_cypher(monkeypatch)
        # Strip // comments first: prose explaining the ordering mentions "ORDER BY"
        # and would otherwise be matched instead of the clause itself.
        code = "\n".join(
            line for line in query.splitlines() if not line.strip().startswith("//")
        )
        assert "$comparable_pairs" in code
        assert code.index("$comparable_pairs") < code.index("ORDER BY")
        assert code.index("$comparable_pairs") < code.index("LIMIT")

    def test_cypher_pair_keys_match_the_python_allow_list(self, monkeypatch):
        """The two implementations of the rule must not drift apart."""
        from app.services.graph_builder import _COMPARABLE_PAIR_KEYS, _role_pair_reason
        _, params = self._captured_cypher(monkeypatch)
        assert params["comparable_pairs"] == _COMPARABLE_PAIR_KEYS
        for key in _COMPARABLE_PAIR_KEYS:
            a, b = key.split("|")
            assert _role_pair_reason(a, b) is not None, key


class TestConflictRowDedup:
    def test_same_values_across_document_pairs_collapse(self):
        """One logical conflict appearing in three doc pairings must render once."""
        contradictions = [
            {"doc1_id": "a", "doc2_id": "b", "value1": "EGP 176,700", "value2": "EGP 155,000",
             "comparison_reason": "invoice total vs invoice subtotal (VAT reconciliation)"},
            {"doc1_id": "a", "doc2_id": "c", "value1": "EGP 176,700.00", "value2": "EGP 155,000",
             "comparison_reason": "invoice total vs invoice subtotal (VAT reconciliation)"},
            {"doc1_id": "b", "doc2_id": "c", "value1": "EGP 155,000", "value2": "EGP 176,700",
             "comparison_reason": "invoice total vs invoice subtotal (VAT reconciliation)"},
        ]
        rows = rp._contradictions_to_rows(contradictions)
        assert len(rows) == 1

    def test_different_value_pairs_are_kept_separately(self):
        """200k-vs-180k and 200k-vs-150k are distinct findings, not duplicates."""
        reason = "contract total vs invoice's stated contract reference"
        contradictions = [
            {"doc1_id": "a", "doc2_id": "b", "value1": "EGP 200,000", "value2": "EGP 180,000",
             "comparison_reason": reason},
            {"doc1_id": "a", "doc2_id": "c", "value1": "EGP 200,000", "value2": "EGP 150,000",
             "comparison_reason": reason},
        ]
        rows = rp._contradictions_to_rows(contradictions)
        assert len(rows) == 2


# ── 3c: signatory authority ──────────────────────────────────────────────────

class TestSigningAuthorityDerivation:
    """
    The model returned signing_authority_level="other" for a person whose title it had
    correctly read as "Chairman", which made an authority comparison meaningless.
    """

    def _auth(self, title):
        from app.services.entity_extractor import _authority_from_title
        return _authority_from_title(title)

    def test_titles_map_to_levels(self):
        assert self._auth("Chairman") == "chairman"
        assert self._auth("Chairman - Delta Trading Group S.A.E.") == "chairman"
        assert self._auth("CEO - Nile Financial Consulting Co.") == "ceo"
        assert self._auth("Chief Financial Officer") == "cfo"
        assert self._auth("Finance Director") == "director"

    def test_arabic_titles_map(self):
        assert self._auth("رئيس مجلس الادارة") == "chairman"

    def test_unknown_title_yields_none(self):
        assert self._auth("Witness") is None
        assert self._auth(None) is None
        assert self._auth("") is None

    def test_chairman_wins_over_a_trailing_manager_word(self):
        """Most-senior-first ordering: 'Chairman & General Manager' is a chairman."""
        assert self._auth("Chairman & General Manager") == "chairman"


class TestSignatoryMismatchQuery:
    """Fires only for an amendment signed below the original signer's authority."""

    def _captured(self, monkeypatch, people):
        import app.services.graph_builder as gb
        captured = {}

        class _FakeSession:
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def run(self, query, **kwargs):
                captured["query"] = query
                return list(people)

        class _FakeDriver:
            def session(self, **kwargs): return _FakeSession()

        monkeypatch.setattr(gb, "_run_with_reconnect", lambda fn: fn(_FakeDriver()))
        rows = gb.find_signatory_mismatches(["d1", "d2"])
        return captured["query"], rows

    def _person(self, name, authority, signed, doc="d1"):
        return {
            "name": name, "title": authority, "authority": authority,
            "signed": signed, "doc_id": doc, "doc_name": f"{doc}.pdf", "page": 1,
        }

    def test_query_only_fetches_people_who_signed_something(self, monkeypatch):
        query, _ = self._captured(monkeypatch, [])
        assert "size(coalesce(p.documents_signed, [])) > 0" in query
        assert "p.entity_type = 'person'" in query

    def test_normal_approval_chain_is_silent(self, monkeypatch):
        """QS certifies, engineer approves, PM approves — different docs, no amendment."""
        people = [
            self._person("Rania Hassan", "manager", ["Payment Certificate PC-2025-007"]),
            self._person("Khaled El Masry", "chairman", ["Board Resolution BR-2024-11"]),
            self._person("Samir Awad", "director", ["QS Report QS-2025-03"]),
            self._person("Ahmed El-Basiony", "manager", ["Inspection sign-off"]),
        ]
        _, rows = self._captured(monkeypatch, people)
        assert rows == []

    def test_amendment_below_original_authority_fires(self, monkeypatch):
        people = [
            self._person("Khaled El Masry", "chairman", ["Contract BLD-2024-019"]),
            self._person("Mona Adel", "cfo", ["Amendment 2 to Contract BLD-2024-019"]),
        ]
        _, rows = self._captured(monkeypatch, people)
        assert len(rows) == 1
        assert rows[0]["name1"] == "Khaled El Masry"   # original signer
        assert rows[0]["name2"] == "Mona Adel"         # amendment signer (lower)
        assert rows[0]["authority_gap"] >= 1

    def test_amendment_at_or_above_original_authority_is_silent(self, monkeypatch):
        people = [
            self._person("Mona Adel", "cfo", ["Contract BLD-2024-019"]),
            self._person("Khaled El Masry", "chairman", ["Amendment 1 to Contract BLD-2024-019"]),
        ]
        _, rows = self._captured(monkeypatch, people)
        assert rows == []

    def test_amendment_without_shared_reference_is_silent(self, monkeypatch):
        people = [
            self._person("Khaled El Masry", "chairman", ["Contract BLD-2024-019"]),
            self._person("Mona Adel", "cfo", ["Amendment to some other agreement"]),
        ]
        _, rows = self._captured(monkeypatch, people)
        assert rows == []

    def test_empty_doc_ids_short_circuits(self):
        from app.services.graph_builder import find_signatory_mismatches
        assert find_signatory_mismatches([]) == []


class TestIsAmendmentOf:
    def _f(self, cand, base):
        from app.services.graph_builder import _is_amendment_of
        return _is_amendment_of(cand, base)

    def test_shared_ref_plus_keyword(self):
        assert self._f("Amendment 2 to BLD-2024-019", "Contract BLD-2024-019") is True

    def test_keyword_without_link_is_false(self):
        assert self._f("Amendment 2 to CTR-2099-001", "Contract BLD-2024-019") is False

    def test_no_keyword_is_false(self):
        assert self._f("Contract BLD-2024-019 schedule", "Contract BLD-2024-019") is False

    def test_same_string_is_false(self):
        assert self._f("Contract BLD-2024-019", "Contract BLD-2024-019") is False


# ── 3d: rate-like roles compare without needing a shared anchor ──────────────

class TestAnchorlessRatePairs:
    def _captured_cypher(self, monkeypatch):
        import app.services.graph_builder as gb
        captured = {}

        class _FakeSession:
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def run(self, query, **kwargs):
                captured["query"] = query
                captured["params"] = kwargs
                return []

        class _FakeDriver:
            def session(self, **kwargs): return _FakeSession()

        monkeypatch.setattr(gb, "_run_with_reconnect", lambda fn: fn(_FakeDriver()))
        gb.find_contradictions(["d1", "d2"])
        return captured["query"], captured["params"]

    def test_anchor_is_optional_for_rate_roles(self, monkeypatch):
        """
        A retainer rate change went undetected because the two figures never shared an
        anchor edge — extraction simply had not linked both to the contract node.
        """
        query, params = self._captured_cypher(monkeypatch)
        assert "OPTIONAL MATCH (e1)-[:RELATES]-(anchor:Entity)-[:RELATES]-(e2)" in query
        assert "$anchorless_roles" in query
        assert "retainer" in params["anchorless_roles"]
        assert "milestone_scheduled" in params["anchorless_roles"]

    def test_totals_still_require_an_anchor(self):
        """Two unrelated invoice totals must not be compared just for existing."""
        from app.services.graph_builder import _ANCHORLESS_SAME_ROLE_PAIRS
        assert "total_invoice" not in _ANCHORLESS_SAME_ROLE_PAIRS
        assert "total_contract_value" not in _ANCHORLESS_SAME_ROLE_PAIRS
