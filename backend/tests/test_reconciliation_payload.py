"""Reconciliation payload: structured values are authoritative; gap-fill is
label-grounded (no positional guessing) and never fabricates from unlabeled prose."""
from app.models.schemas import Finding, ReconciliationSnapshot
from app.services.reconciliation_payload import (
    _fill_snapshot_gaps,
    _label_grounded_amounts,
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
