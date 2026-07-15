"""Finding deduplication helpers: topic bucketing, numeric/ref signatures, and preference."""
import re

from app.models.schemas import Finding
from app.utils.arabic_normalizer import extract_amounts
from app.utils.money_parse import parse_monetary_amount
from app.utils.text_amount_scan import amounts_from_text_scan


def _finding_topic_bucket(f: Finding) -> str:
    """Bucket findings for dedup. Bank/contract/invoice reconciliation stories share one bucket."""
    blob = (f.title + " " + (f.description or "")[:800]).lower()

    bank_contract_gap_kw = (
        "overpayment",
        "exceeds contract",
        "exceeds the contract",
        "exceeds invoice",
        "exceeds both",
        "paid more",
        "overpaid",
        "contract value",
        "contract total",
        "contract cap",
        "bank paid",
        "total paid",
        "versus contract",
        "vs contract",
        "cumulative",
        "unauthorized",
        "overrun",
        "above contract",
        "beyond the contract",
        "authorized value",
        "variance",
    )
    if any(k in blob for k in bank_contract_gap_kw):
        return "bank_contract_gap"

    # Invoice vs bank vs contract in one narrative → same bucket as bank–contract gap
    invoice_markers = ("invoice", "inv-", "invoicing", "billed", "subtotal")
    payment_markers = (
        "contract",
        "bank",
        "paid",
        "payment",
        "payable",
        "disbursement",
        "transfer",
        "milestone",
        "advance",
        "overpayment",
        "exceeds",
        "debit",
        "credit",
    )
    if any(i in blob for i in invoice_markers) and any(p in blob for p in payment_markers):
        return "bank_contract_gap"

    rules: list[tuple[str, tuple[str, ...]]] = [
        ("retainer", ("retainer", "march", "monthly", "article 2")),
        ("invoice", ("invoice", "inv-", "billed", "subtotal", "invoicing")),
        ("schedule", ("schedule", "milestone", "due date", "payment date", "payment schedule")),
        ("vat", ("vat", "tax")),
    ]
    for bucket, kws in rules:
        if any(k in blob for k in kws):
            return bucket
    return "general"


# CTR-2024-044 / INV-2025-001 style anchors for dedup (title + description only)
_REF_TOKEN_RE = re.compile(
    r"\b(?:CTR|INV|CNT|PO|CON|SI)[-\s]?\d{2,4}[-\s]?\d{2,}\b",
    re.IGNORECASE,
)

_MERGE_PAIR_TOPICS = frozenset({"bank_contract_gap", "invoice"})


def _topics_compatible_for_dedup(a: str, b: str) -> bool:
    if a == b:
        return True
    return a in _MERGE_PAIR_TOPICS and b in _MERGE_PAIR_TOPICS


def _finding_ref_signature(f: Finding) -> set[str]:
    text = f"{f.title}\n{f.description or ''}"
    out: set[str] = set()
    for m in _REF_TOKEN_RE.finditer(text):
        tok = re.sub(r"\s+", "", m.group(0).upper())
        out.add(tok)
    return out


def _amount_sets_jaccard(a: set[float], b: set[float]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def _numeric_signature_overlap(
    f_sig: set[float],
    p_sig: set[float],
    f_refs: set[str],
    p_refs: set[str],
) -> bool:
    """Intersection, Jaccard, or shared contract/invoice ref + partial numeric overlap."""
    if not f_sig and not p_sig:
        return False
    if not f_sig or not p_sig:
        return bool(f_refs & p_refs)
    if f_sig & p_sig:
        return True
    j = _amount_sets_jaccard(f_sig, p_sig)
    if j >= 0.25:
        return True
    if f_refs & p_refs and j >= 0.08:
        return True
    # Same contract/invoice ref and a small combined set of headline amounts → one audit story
    if f_refs & p_refs and len(f_sig | p_sig) <= 8 and min(len(f_sig), len(p_sig)) >= 1:
        return True
    return False


# Dedup signatures must not use evidence[]: snippets repeat balance/tax lines with huge varying
# amounts (326k, 750k, …) so each finding would get a different tuple and duplicates survive.
_SIGNATURE_AMOUNT_CAP = 2_000_000.0


def _amount_for_dedup_signature(p: float) -> bool:
    """
    Keep only headline-scale figures for clustering the same audit story.
    """
    if p < 100.0 or p > _SIGNATURE_AMOUNT_CAP:
        return False
    return True


def _canonical_amount(raw_value: float) -> float:
    """
    Normalize an amount to a coarse bucket for dedup matching.
    Values < 10k round to nearest 100, >= 10k round to nearest 1000.
    This absorbs minor OCR/rounding variations while keeping distinct amounts separate.
    """
    if raw_value < 10_000:
        return round(raw_value / 100) * 100
    return round(raw_value / 1_000) * 1_000


def _finding_numeric_signature(f: Finding) -> tuple[float, ...]:
    """
    Stable canonicalized amount set from title + description only.
    Evidence columns are excluded to avoid noisy balance/running-total lines
    creating unique signatures for what is actually the same audit story.
    """
    text = f"{f.title}\n{f.description or ''}"
    nums: list[float] = []
    for amt in extract_amounts(text):
        raw = (amt.get("raw") or "").strip()
        p = parse_monetary_amount(raw)
        if p is None and amt.get("value") is not None:
            try:
                p = float(amt["value"])
            except (TypeError, ValueError):
                p = None
        if p is not None and _amount_for_dedup_signature(p):
            nums.append(_canonical_amount(p))
    for p in amounts_from_text_scan(text, min_value=1000):
        if _amount_for_dedup_signature(p):
            nums.append(_canonical_amount(p))
    return tuple(sorted(set(nums)))


def _prefer_finding(candidate: Finding, incumbent: Finding) -> bool:
    """True if candidate should replace incumbent (severity first, then confidence)."""
    sev_rank = {"critical": 3, "warning": 2, "ok": 1}
    cr = sev_rank.get(candidate.severity, 0)
    ir = sev_rank.get(incumbent.severity, 0)
    if cr > ir:
        return True
    if cr < ir:
        return False
    return candidate.confidence_score > incumbent.confidence_score


def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    """
    Deduplicate findings into a single clearest story per topic/amount/reference cluster.

    Pre-computes each finding's dedup signature once, then runs the O(n^2) pairwise
    comparison used across the codebase (same-topic, doc-overlap, and numeric/ref
    signature overlap all being independent axes of "this is the same audit story").
    """
    def _dedup_signature(f: Finding) -> tuple[str, set, set, set]:
        return (
            _finding_topic_bucket(f),
            set(_finding_numeric_signature(f)),
            set(filter(None, [f.source_doc_id, f.conflicting_doc_id])),
            _finding_ref_signature(f),
        )

    finding_sigs = [_dedup_signature(f) for f in findings]

    deduped: list[Finding] = []
    deduped_sigs: list[tuple[str, set, set, set]] = []
    for f, (f_topic, f_sig, f_docs, f_refs) in zip(findings, finding_sigs):
        merged = False
        for idx, (prev, (p_topic, p_sig, p_docs, p_refs)) in enumerate(
            zip(deduped, deduped_sigs)
        ):
            is_dup = False

            if (
                _topics_compatible_for_dedup(f_topic, p_topic)
                and f_topic != "general"
                and p_topic != "general"
            ):
                doc_overlap = not f_docs or not p_docs or bool(f_docs & p_docs)
                sig_overlap = False

                if f_sig and p_sig:
                    sig_overlap = _numeric_signature_overlap(f_sig, p_sig, f_refs, p_refs)
                elif not f_sig and not p_sig:
                    if f.title[:30].lower() == prev.title[:30].lower():
                        sig_overlap = True
                elif f_refs & p_refs and (not f_sig or not p_sig):
                    sig_overlap = True

                if doc_overlap and sig_overlap:
                    is_dup = True

            elif f_docs == p_docs and f_sig == p_sig and f_sig:
                is_dup = True
            elif f_docs == p_docs and not f_sig and not p_sig and f.title[:40].lower() == prev.title[:40].lower():
                is_dup = True

            if is_dup:
                if _prefer_finding(f, prev):
                    deduped[idx] = f
                    deduped_sigs[idx] = (f_topic, f_sig, f_docs, f_refs)
                merged = True
                break

        if not merged:
            deduped.append(f)
            deduped_sigs.append((f_topic, f_sig, f_docs, f_refs))

    return deduped
