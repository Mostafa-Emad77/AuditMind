"""Reconciliation payload: structured values are authoritative; gap-fill is
label-grounded (no positional guessing) and never fabricates from unlabeled prose."""
from app.models.schemas import Finding, ReconciliationSnapshot
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

def _amount_finding(sev: str, title: str, desc: str) -> Finding:
    return Finding(
        severity=sev, title=title, description=desc, confidence_score=0.9,
        source_doc_id="docA", conflicting_doc_id="docB",
    )


def test_amount_findings_become_conflict_rows():
    findings = [
        _amount_finding(
            "critical", "M-3 payment exceeds certified amount",
            "Bank paid EGP 37,006,052 against certified EGP 36,465,379.",
        ),
        _amount_finding(
            "warning", "Bilingual due date conflict",
            "The Arabic text says 30 days; the English says 45 days.",  # no 2 amounts
        ),
    ]
    rows = _findings_to_conflict_rows(findings)
    assert len(rows) == 1
    assert rows[0].severity == "critical"
    assert rows[0].doc_a_id == "docA" and rows[0].doc_b_id == "docB"


def test_merge_keeps_findings_rows_and_dedupes_graph_rows():
    finding_rows = _findings_to_conflict_rows([
        _amount_finding(
            "critical", "Overpayment", "Bank paid EGP 37,006,052 vs certified EGP 36,465,379.",
        )
    ])
    from app.models.schemas import EntityConflictRow
    graph_dup = EntityConflictRow(
        entity_label="x", doc_a_id="d1", doc_a_value="37,006,052.00",
        doc_b_id="d2", doc_b_value="36,465,379.00", severity="critical",
    )
    graph_new = EntityConflictRow(
        entity_label="y", doc_a_id="d1", doc_a_value="1,000",
        doc_b_id="d2", doc_b_value="2,000", severity="warning",
    )
    merged = _merge_conflict_rows(finding_rows, [graph_dup, graph_new])
    # 1 finding row + the non-duplicate graph row only
    assert len(merged) == 2
    assert merged[0].conflict_type == "amount finding"
