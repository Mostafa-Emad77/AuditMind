"""Post-generation guard: findings must not cite internal UUID identifiers that
are not quoted verbatim in their own evidence (BUG 1 — hallucinated contract IDs)."""
from app.models.schemas import Finding
from app.agents.cross_checker.validation import drop_hallucinated_identifier_findings

_UUID_A = "8fc1c2b3-2b1b-450e-bb03-dbbc5921d5e2"
_UUID_B = "4b693574-ea86-41a6-b0f0-9c4717d92c12"


def _f(title: str, description: str, evidence: list[str]) -> Finding:
    return Finding(
        severity="critical", title=title, description=description,
        confidence_score=0.9, evidence=evidence,
    )


def test_drops_finding_citing_uuid_absent_from_evidence():
    findings = [
        _f(
            "Checklist requires cross-referencing two contract IDs",
            f"Contract {_UUID_A} is missing from the audit trail; expected alongside {_UUID_B}.",
            ["contract.pdf, page 1: The contract number is BLD-2024-019."],
        ),
    ]
    kept, dropped = drop_hallucinated_identifier_findings(findings)
    assert dropped == 1
    assert kept == []


def test_keeps_finding_when_uuid_actually_appears_in_evidence():
    findings = [
        _f(
            "Reference mismatch",
            f"Document references {_UUID_A}.",
            [f"source.pdf, page 2: external ref {_UUID_A} was quoted verbatim here."],
        ),
    ]
    kept, dropped = drop_hallucinated_identifier_findings(findings)
    assert dropped == 0
    assert len(kept) == 1


def test_keeps_normal_findings_untouched():
    findings = [
        _f(
            "M-3 payment exceeds certified amount",
            "Bank paid EGP 37,006,052 against certified EGP 36,465,379.",
            ["bank.pdf p.3: 37,006,052", "cert.pdf p.1: 36,465,379"],
        ),
    ]
    kept, dropped = drop_hallucinated_identifier_findings(findings)
    assert dropped == 0
    assert kept == findings


def test_human_readable_reference_is_not_a_uuid():
    findings = [
        _f(
            "Contract BLD-2024-019 total mismatch",
            "Contract BLD-2024-019 states EGP 40,000,000 but invoice claims EGP 41,000,000.",
            ["contract.pdf p.1: EGP 40,000,000"],
        ),
    ]
    kept, dropped = drop_hallucinated_identifier_findings(findings)
    assert dropped == 0
    assert len(kept) == 1
