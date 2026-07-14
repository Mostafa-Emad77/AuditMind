"""
Stable, document-independent signature for a Finding.

Used by the false-positive feedback loop: when a reviewer dismisses a finding as a
false positive, we store its signature in a global set; the cross-checker consults
that set on later runs and suppresses matching candidates. The signature must be
stable across audits (different doc_ids, slightly different wording) so it is built
from the finding's *topic* plus its *canonical headline amounts* — never from
doc_ids, pages, or evidence text.
"""
from typing import Any

from app.utils.arabic_normalizer import extract_amounts
from app.utils.money_parse import parse_monetary_amount

# Coarse topic buckets keyed off keywords in the title/description. Mirrors the
# spirit of the cross-checker's dedup buckets but is intentionally self-contained
# (no imports from the heavy cross_checker module).
_TOPIC_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("overpayment", ("overpay", "exceeds", "overrun", "above contract", "beyond the contract", "variance")),
    ("bank_recon", ("bank", "paid", "payment", "transfer", "disburse", "remittance", "debit", "credit")),
    ("invoice", ("invoice", "billed", "billing", "subtotal")),
    ("schedule", ("schedule", "milestone", "due date", "payment date", "installment")),
    ("vat", ("vat", "tax")),
    ("party", ("party", "vendor", "signatory", "name mismatch", "counterparty")),
    ("date", ("date inconsistency", "dated", "expiry", "effective date")),
]

_SIGNATURE_AMOUNT_CAP = 2_000_000.0


def _topic(text: str) -> str:
    for bucket, kws in _TOPIC_KEYWORDS:
        if any(k in text for k in kws):
            return bucket
    return "general"


def _canonical_amount(value: float) -> int:
    """Round to a coarse bucket so minor OCR/rounding variations collapse together."""
    if value < 10_000:
        return int(round(value / 100) * 100)
    return int(round(value / 1_000) * 1_000)


def _headline_amounts(text: str) -> list[int]:
    nums: list[int] = []
    seen: set[int] = set()
    for am in extract_amounts(text):
        raw = (am.get("raw") or "").strip()
        p = parse_monetary_amount(raw)
        if p is None and am.get("value") is not None:
            try:
                p = float(am["value"])
            except (TypeError, ValueError):
                p = None
        if p is None or p < 100.0 or p > _SIGNATURE_AMOUNT_CAP:
            continue
        c = _canonical_amount(p)
        if c not in seen:
            seen.add(c)
            nums.append(c)
    return sorted(nums)


def finding_signature(finding: Any) -> str:
    """
    Build a stable signature string for a finding.

    Shape: ``"<topic>|<amount1,amount2,...>"`` when the finding carries headline
    amounts; otherwise ``"<topic>|t:<title-prefix>"`` so text-only findings can
    still be suppressed. ``finding`` may be a Pydantic ``Finding`` or any object
    exposing ``title`` / ``description``.
    """
    title = str(getattr(finding, "title", "") or "")
    description = str(getattr(finding, "description", "") or "")
    text = f"{title}\n{description}".lower()

    topic = _topic(text)
    amounts = _headline_amounts(text)
    if amounts:
        return f"{topic}|{','.join(str(a) for a in amounts)}"
    return f"{topic}|t:{title.strip().lower()[:40]}"
