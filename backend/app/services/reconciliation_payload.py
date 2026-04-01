"""Build reconciliation snapshot and entity conflict rows for AuditReport."""
from __future__ import annotations

import logging
from typing import Literal, Optional

from app.config import get_settings
from app.models.schemas import (
    DocumentMeta,
    EntityConflictRow,
    Finding,
    ReconciliationSnapshot,
)
from app.services.graph_builder import find_contradictions, get_driver
from app.utils.money_parse import parse_monetary_amount
from app.utils.text_amount_scan import amounts_from_text_scan

logger = logging.getLogger(__name__)

# Heuristic: amounts above this are "headline" totals for max() picks
_MIN_HEADLINE = 1_000.0


def _fetch_amount_rows(doc_ids: list[str]) -> list[dict]:
    if not doc_ids:
        return []
    settings = get_settings()

    def _run():
        with get_driver().session(database=settings.neo4j_database) as session:
            result = session.run(
                """
                MATCH (e:Entity)-[:FOUND_IN]->(d:Document)
                WHERE d.doc_id IN $doc_ids AND e.entity_type = 'amount'
                RETURN d.doc_id AS doc_id,
                       d.doc_type AS doc_type,
                       d.filename AS filename,
                       coalesce(e.amount_role, 'unknown') AS amount_role,
                       e.normalized_value AS normalized_value,
                       e.value AS value,
                       e.entity_id AS entity_id,
                       coalesce(e.source_page, 0) AS source_page
                """,
                doc_ids=doc_ids,
            )
            return [dict(r) for r in result]

    try:
        return _run()
    except Exception as e:
        logger.warning("Neo4j amount fetch failed: %s", e)
        return []


def _parse_row_amount(row: dict) -> Optional[float]:
    raw = row.get("normalized_value") or row.get("value") or ""
    return parse_monetary_amount(str(raw).strip()) if raw else None


def _bank_round_key(val: float) -> float:
    return round(val, 2)


def _aggregate_snapshot(rows: list[dict], documents: list[DocumentMeta]) -> ReconciliationSnapshot:
    doc_type_by_id = {d.doc_id: d.doc_type for d in documents}
    contract_total: Optional[float] = None
    invoice_total: Optional[float] = None
    bank_line_keys: set[tuple] = set()
    bank_sum = 0.0
    bank_any = False
    bank_deduped = False

    contract_candidates: list[float] = []
    invoice_candidates: list[float] = []
    milestone_sums: list[float] = []

    for row in rows:
        dt = doc_type_by_id.get(row.get("doc_id"), "unknown")
        role = (row.get("amount_role") or "unknown").strip().lower()
        val = _parse_row_amount(row)
        if val is None or val < 1:
            continue

        if dt == "contract":
            if role == "total_contract_value":
                contract_candidates.append(val)
            elif role == "milestone_scheduled":
                milestone_sums.append(val)
            elif role not in ("opening_balance", "closing_balance", "unknown"):
                if val >= _MIN_HEADLINE:
                    contract_candidates.append(val)
            elif role == "unknown" and val >= _MIN_HEADLINE:
                contract_candidates.append(val)
        elif dt == "invoice":
            if role == "total_invoice":
                invoice_candidates.append(val)
            elif val >= _MIN_HEADLINE:
                invoice_candidates.append(val)
        elif dt == "bank_statement":
            if role in ("opening_balance", "closing_balance"):
                continue
            include_bank = role == "single_payment" or (role == "unknown" and val >= 100)
            if not include_bank:
                continue
            doc_id = str(row.get("doc_id") or "")
            page = int(row.get("source_page") or 0)
            # One logical line per doc/page/amount: duplicate Entity nodes (re-chunking/OCR)
            # share the same page+value but have different entity_ids.
            bkey: tuple = ("line", doc_id, page, _bank_round_key(val))
            if bkey in bank_line_keys:
                bank_deduped = True
                continue
            bank_line_keys.add(bkey)
            bank_sum += val
            bank_any = True

    if contract_candidates:
        contract_total = max(contract_candidates)
    elif milestone_sums:
        contract_total = sum(milestone_sums)

    if invoice_candidates:
        invoice_total = max(invoice_candidates)

    bank_paid = bank_sum if bank_any else None

    variance: Optional[float] = None
    notes_parts: list[str] = []
    if bank_deduped:
        notes_parts.append("Bank total deduplicated (duplicate amount entities in graph).")
    if contract_total is not None and bank_paid is not None:
        variance = round(bank_paid - contract_total, 2)
    elif contract_total is None and bank_paid is not None:
        notes_parts.append("Contract total not isolated in graph; bank total shown.")
    elif bank_paid is None:
        notes_parts.append("Bank payment total not aggregated from graph.")

    return ReconciliationSnapshot(
        currency="EGP",
        contract_total=round(contract_total, 2) if contract_total is not None else None,
        invoice_total=round(invoice_total, 2) if invoice_total is not None else None,
        bank_paid_total=round(bank_paid, 2) if bank_paid is not None else None,
        variance_vs_contract=variance,
        notes=" ".join(notes_parts) if notes_parts else None,
    )


def _strip_dominant_outliers(unique_desc: list[float]) -> list[float]:
    """Remove one huge spike so fallback does not treat duplicated graph totals as 'bank'."""
    s = list(unique_desc)
    while len(s) >= 2 and s[0] > 15 * s[1]:
        s = s[1:]
    return s


def _fallback_snapshot_from_findings(findings: list[Finding]) -> Optional[ReconciliationSnapshot]:
    """Extract plausible headline numbers from finding text when graph data is thin."""
    text = " ".join(
        f.title + " " + f.description for f in findings if f.severity == "critical"
    )
    nums = amounts_from_text_scan(text, min_value=_MIN_HEADLINE)
    if len(nums) < 2:
        return None
    nums_sorted = _strip_dominant_outliers(sorted(set(nums), reverse=True))
    if len(nums_sorted) < 2:
        return None
    # Heuristic: largest = bank or contract mix; often 226700, 200000, 176700 in demo
    bank_guess = nums_sorted[0] if nums_sorted else None
    contract_guess = next((n for n in nums_sorted if n < bank_guess), None) if bank_guess else None
    invoice_guess = next((n for n in nums_sorted if n not in (bank_guess, contract_guess)), None)
    if bank_guess and contract_guess and bank_guess > contract_guess:
        return ReconciliationSnapshot(
            currency="EGP",
            contract_total=contract_guess,
            invoice_total=invoice_guess if invoice_guess and invoice_guess < bank_guess else None,
            bank_paid_total=bank_guess,
            variance_vs_contract=round(bank_guess - contract_guess, 2),
            notes="Partially inferred from finding text (graph data incomplete).",
        )
    return None


def _should_override_bank_with_findings(
    snapshot: ReconciliationSnapshot,
    fb: Optional[ReconciliationSnapshot],
) -> bool:
    if fb is None or fb.bank_paid_total is None:
        return False
    ct = snapshot.contract_total
    bp = snapshot.bank_paid_total
    if ct is None or bp is None:
        return False
    if ct >= 2_000_000:
        return False
    if bp <= max(ct * 8, 500_000):
        return False
    return fb.bank_paid_total < bp * 0.5


def _merge_bank_from_fallback(
    snapshot: ReconciliationSnapshot,
    fb: ReconciliationSnapshot,
) -> ReconciliationSnapshot:
    extra = "Bank total adjusted using critical finding amounts (graph sum implausible)."
    parts = [p for p in (snapshot.notes, fb.notes, extra) if p]
    notes = " ".join(parts).strip()
    ct = snapshot.contract_total
    bank = fb.bank_paid_total
    var: Optional[float] = None
    if ct is not None and bank is not None:
        var = round(bank - ct, 2)
    return ReconciliationSnapshot(
        currency=snapshot.currency,
        contract_total=snapshot.contract_total,
        invoice_total=snapshot.invoice_total,
        bank_paid_total=round(bank, 2) if bank is not None else None,
        variance_vs_contract=var,
        notes=notes or None,
    )


def _contradictions_to_rows(contradictions: list[dict]) -> list[EntityConflictRow]:
    rows: list[EntityConflictRow] = []
    seen: set[tuple] = set()
    for c in contradictions:
        d1 = c.get("doc1_id") or ""
        d2 = c.get("doc2_id") or ""
        v1 = str(c.get("value1") or c.get("norm1") or "").strip()
        v2 = str(c.get("value2") or c.get("norm2") or "").strip()
        key = (d1, d2, v1, v2)
        if key in seen or not d1 or not d2 or d1 == d2:
            continue
        seen.add(key)
        anchor = str(c.get("anchor_value") or "").strip() or None
        label = anchor if anchor else "Cross-document amount"
        n1 = parse_monetary_amount(v1) or 0.0
        n2 = parse_monetary_amount(v2) or 0.0
        diff = abs(n1 - n2)
        mx = max(n1, n2, 1.0)
        rel = diff / mx
        sev: Literal["critical", "warning"] = "critical" if diff >= 5000 or rel >= 0.05 else "warning"
        rows.append(
            EntityConflictRow(
                entity_label=label[:200],
                doc_a_id=d1,
                doc_a_value=v1[:500],
                doc_b_id=d2,
                doc_b_value=v2[:500],
                severity=sev,
                conflict_type="amount",
                anchor_hint=anchor,
            )
        )
    return rows[:30]


def _snapshot_has_numbers(s: ReconciliationSnapshot) -> bool:
    return any(
        x is not None for x in (s.contract_total, s.invoice_total, s.bank_paid_total)
    )


def _normalize_documents(documents: list) -> list[DocumentMeta]:
    out: list[DocumentMeta] = []
    for d in documents:
        if isinstance(d, DocumentMeta):
            out.append(d)
        elif isinstance(d, dict):
            try:
                out.append(DocumentMeta.model_validate(d))
            except Exception:
                continue
    return out


def build_reconciliation_payload(
    documents: list,
    findings: list[Finding],
) -> tuple[Optional[ReconciliationSnapshot], list[EntityConflictRow]]:
    """
    Build structured reconciliation snapshot and conflict-only rows for the report.
    Uses Neo4j entity amounts + contradiction pairs, with a findings-text fallback.
    """
    documents = _normalize_documents(documents)
    doc_ids = [d.doc_id for d in documents]
    amount_rows = _fetch_amount_rows(doc_ids)
    snapshot = _aggregate_snapshot(amount_rows, documents)
    fb = _fallback_snapshot_from_findings(findings) if findings else None

    if not _snapshot_has_numbers(snapshot) and fb:
        snapshot = fb
    elif fb and _should_override_bank_with_findings(snapshot, fb):
        snapshot = _merge_bank_from_fallback(snapshot, fb)

    if not _snapshot_has_numbers(snapshot):
        snapshot = None

    conflicts = _contradictions_to_rows(find_contradictions(doc_ids))
    return snapshot, conflicts
