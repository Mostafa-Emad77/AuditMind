"""Reconciliation payload: structured values are authoritative; gap-fill is
label-grounded (no positional guessing) and never fabricates from unlabeled prose."""
from app.models.schemas import (
    DocumentMeta,
    EntityConflictRow,
    Finding,
    ReconciliationSnapshot,
)
from app.services.reconciliation_payload import (
    _fill_snapshot_gaps,
    _findings_to_conflict_rows,
    _label_grounded_amounts,
    _merge_conflict_rows,
    _role_for_window,
)


def _critical(title: str, desc: str) -> Finding:
    return Finding(severity="critical", title=title, description=desc, confidence_score=0.9)


def test_role_for_window_picks_nearest_label():
    assert _role_for_window("contract value vs bank paid ") == "bank_paid_total"
    assert _role_for_window("the total invoice ") == "invoice_total"
    assert _role_for_window("agreed contract ") == "contract_total"
    assert _role_for_window("overpayment of ") is None


def test_label_grounded_amounts_assigns_by_adjacent_label():
    f = _critical(
        "Bank paid more than contract value",
        "Bank paid EGP 226,700 vs contract EGP 200,000 — overpayment EGP 26,700.",
    )
    got = _label_grounded_amounts([f])
    assert got.get("bank_paid_total") == 226700.0
    assert got.get("contract_total") == 200000.0
    # The unlabeled "overpayment" amount must not be assigned to any field.
    assert 26700.0 not in got.values()


def test_label_grounded_ignores_unlabeled_amounts():
    f = _critical("Discrepancy", "Figures of EGP 50,000 and EGP 70,000 were noted.")
    assert _label_grounded_amounts([f]) == {}


def test_label_grounded_skips_non_critical():
    f = Finding(
        severity="warning",
        title="contract EGP 200,000",
        description="bank paid EGP 226,700",
        confidence_score=0.9,
    )
    assert _label_grounded_amounts([f]) == {}


def test_fill_gaps_never_overrides_structured_values():
    snap = ReconciliationSnapshot(
        currency="EGP", contract_total=200000.0, invoice_total=None, bank_paid_total=None,
    )
    inferred = {"contract_total": 999999.0, "bank_paid_total": 226700.0}
    out = _fill_snapshot_gaps(snap, inferred)
    assert out.contract_total == 200000.0       # structured value kept
    assert out.bank_paid_total == 226700.0       # empty field gap-filled
    assert out.variance_vs_contract == 26700.0   # variance recomputed
    assert "Inferred" in (out.notes or "")


def test_fill_gaps_noop_when_nothing_to_fill():
    snap = ReconciliationSnapshot(currency="EGP", contract_total=1.0)
    assert _fill_snapshot_gaps(snap, {}) is snap


def test_fill_gaps_never_reintroduces_withheld_bank_total():
    """When extraction is incomplete the bank total is withheld and must stay withheld."""
    snap = ReconciliationSnapshot(
        currency="EGP", contract_total=200000.0, bank_paid_total=None,
        bank_extraction_incomplete=True,
    )
    out = _fill_snapshot_gaps(snap, {"bank_paid_total": 540673.0})
    assert out.bank_paid_total is None


# ── Reconciliation conflict rows stay consistent with the Findings tab (BUG 3) ──

CONTRACT = DocumentMeta(doc_id="docA", filename="contract.pdf", doc_type="contract")
BANK = DocumentMeta(doc_id="docB", filename="bank.pdf", doc_type="bank_statement")
DOCS = [CONTRACT, BANK]


def _amount_row(doc_id: str, value: str) -> dict:
    return {"doc_id": doc_id, "normalized_value": value, "value": value}


def _amount_finding(sev: str, title: str, desc: str, evidence=None) -> Finding:
    return Finding(
        severity=sev, title=title, description=desc, confidence_score=0.9,
        source_doc_id="docA", conflicting_doc_id="docB", evidence=evidence or [],
    )


def test_conflict_row_values_sit_under_the_document_they_came_from():
    """
    The regression: amounts were sorted by size and assigned positionally, so the
    bank's 226,700 was rendered in the contract's column and the contract's 200,000
    in the bank's. Each value must follow its own document.
    """
    f = _amount_finding(
        "critical", "Bank payments exceed contract total",
        "Contract specifies a total contract value of EGP 200,000.00. The bank statement "
        "records four payments totaling EGP 226,700.00, which exceeds the contract total "
        "by EGP 26,700.00.",
    )
    rows = _findings_to_conflict_rows(
        [f], DOCS,
        [_amount_row("docA", "200000 EGP"), _amount_row("docB", "226700 EGP")],
    )
    assert len(rows) == 1
    r = rows[0]
    assert (r.doc_a_id, r.doc_a_value) == ("docA", "200,000.00")
    assert (r.doc_b_id, r.doc_b_value) == ("docB", "226,700.00")
    assert r.conflict_type == "contract vs bank statement (from finding)"


def test_computed_differences_are_not_treated_as_document_values():
    """26,700 exists in no document — it is the subtraction, not a stated figure."""
    f = _amount_finding(
        "critical", "Overpayment",
        "Contract value EGP 200,000.00 vs bank paid EGP 226,700.00, a gap of EGP 26,700.00.",
    )
    rows = _findings_to_conflict_rows(
        [f], DOCS,
        [_amount_row("docA", "200000 EGP"), _amount_row("docB", "226700 EGP")],
    )
    assert [rows[0].doc_a_value, rows[0].doc_b_value] == ["200,000.00", "226,700.00"]


def test_evidence_grounding_beats_an_ambiguous_graph_match():
    """An amount quoted in an evidence line belongs to that line's document."""
    f = _amount_finding(
        "warning", "Final payment mismatch",
        "Contract lists Final Payment of 35,000.00 EGP. Bank shows 45,370.00 EGP.",
        evidence=[
            "contract.pdf, page 1: Final Payment | 31 Mar 2025 | 35,000.00",
            "bank.pdf, page 1: Final settlement 45,370.00",
        ],
    )
    # Both amounts appear in both documents in the graph — evidence must decide.
    rows = _findings_to_conflict_rows(
        [f], DOCS,
        [_amount_row("docA", "35000 EGP"), _amount_row("docB", "35000 EGP"),
         _amount_row("docA", "45370 EGP"), _amount_row("docB", "45370 EGP")],
    )
    assert (rows[0].doc_a_id, rows[0].doc_a_value) == ("docA", "35,000.00")
    assert (rows[0].doc_b_id, rows[0].doc_b_value) == ("docB", "45,370.00")


def test_unattributable_finding_is_skipped():
    """No row at all beats a row against the wrong document."""
    f = _amount_finding(
        "critical", "Something is off", "Figures of EGP 11,111 and EGP 22,222 disagree.",
    )
    assert _findings_to_conflict_rows([f], DOCS, []) == []


def test_single_document_finding_produces_no_row():
    f = _amount_finding(
        "warning", "Internal inconsistency",
        "Contract states EGP 200,000.00 in one clause and EGP 150,000.00 in another.",
    )
    rows = _findings_to_conflict_rows(
        [f], DOCS,
        [_amount_row("docA", "200000 EGP"), _amount_row("docA", "150000 EGP")],
    )
    assert rows == []


def test_ok_severity_findings_are_ignored():
    f = Finding(severity="ok", title="Aligned", description="EGP 200,000.00 vs EGP 226,700.00",
                confidence_score=0.9, source_doc_id="docA", conflicting_doc_id="docB")
    assert _findings_to_conflict_rows(
        [f], DOCS,
        [_amount_row("docA", "200000 EGP"), _amount_row("docB", "226700 EGP")],
    ) == []


def test_merge_keeps_findings_rows_and_dedupes_graph_rows():
    finding_rows = _findings_to_conflict_rows(
        [_amount_finding("critical", "Overpayment",
                         "Contract value EGP 200,000.00 vs bank paid EGP 226,700.00.")],
        DOCS,
        [_amount_row("docA", "200000 EGP"), _amount_row("docB", "226700 EGP")],
    )
    assert len(finding_rows) == 1
    graph_dup = EntityConflictRow(
        entity_label="x", doc_a_id="d1", doc_a_value="200,000.00",
        doc_b_id="d2", doc_b_value="226,700.00", severity="critical",
    )
    graph_new = EntityConflictRow(
        entity_label="y", doc_a_id="d1", doc_a_value="1,000",
        doc_b_id="d2", doc_b_value="2,000", severity="warning",
    )
    merged = _merge_conflict_rows(finding_rows, [graph_dup, graph_new])
    # 1 finding row + the non-duplicate graph row only
    assert len(merged) == 2
    assert merged[0].conflict_type.endswith("(from finding)")
