"""Build reconciliation snapshot and entity conflict rows for AuditReport."""
from __future__ import annotations

import logging
from typing import Literal, Optional

from app.config import get_settings
from app.models.schemas import (
    BANK_OUTFLOW_ROLES,
    NON_SUMMABLE_ROLES,
    DocumentMeta,
    EntityConflictRow,
    Finding,
    ReconciliationSnapshot,
)
from app.services.graph_builder import find_contradictions, get_driver
from app.utils.money_parse import parse_monetary_amount
from app.utils.arabic_normalizer import extract_amounts

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
                // Anchors used to tell payments to the contract counterparty apart
                // from unrelated outflows (salaries, other vendors).
                OPTIONAL MATCH (e)-[:RELATES]-(anchor:Entity)
                WHERE anchor.entity_type IN ['contract_id', 'invoice_id', 'company', 'person']
                RETURN d.doc_id AS doc_id,
                       d.doc_type AS doc_type,
                       d.filename AS filename,
                       coalesce(e.amount_role, 'unknown') AS amount_role,
                       e.normalized_value AS normalized_value,
                       e.value AS value,
                       e.entity_id AS entity_id,
                       e.txn_ref AS txn_ref,
                       e.txn_date AS txn_date,
                       coalesce(e.source_page, 0) AS source_page,
                       collect(DISTINCT anchor.entity_id) AS anchor_ids
                """,
                doc_ids=doc_ids,
            )
            return [dict(r) for r in result]

    try:
        return _run()
    except Exception as e:
        logger.warning("Neo4j amount fetch failed: %s", e)
        return []


def _contract_anchor_ids(doc_ids: list[str]) -> set[str]:
    """Contract anchors (number, parties): a debit linked to one is a payment on this contract."""
    if not doc_ids:
        return set()
    settings = get_settings()

    def _run():
        with get_driver().session(database=settings.neo4j_database) as session:
            result = session.run(
                """
                MATCH (e:Entity)-[:FOUND_IN]->(d:Document)
                WHERE d.doc_id IN $doc_ids
                  AND d.doc_type = 'contract'
                  AND e.entity_type IN ['contract_id', 'company', 'person']
                RETURN collect(DISTINCT e.entity_id) AS ids
                """,
                doc_ids=doc_ids,
            )
            rec = result.single()
            return set(rec["ids"]) if rec and rec["ids"] else set()

    try:
        return _run()
    except Exception as e:
        logger.warning("Contract anchor fetch failed: %s", e)
        return set()


def _parse_row_amount(row: dict) -> Optional[float]:
    raw = row.get("normalized_value") or row.get("value") or ""
    return parse_monetary_amount(str(raw).strip()) if raw else None


def _bank_round_key(val: float) -> float:
    return round(val, 2)


def _pick_invoice_total(invoice_by_doc: dict[str, dict[str, list[float]]]) -> Optional[float]:
    """Per document, prefer the total that subtotal + VAT supports; else max total, else subtotal."""
    picks: list[float] = []
    for bucket in invoice_by_doc.values():
        totals, subtotals, vats = bucket["total"], bucket["subtotal"], bucket["vat"]
        supported = [
            t for t in totals
            if any(abs(t - (s + v)) <= 1.0 for s in subtotals for v in vats)
        ]
        if supported:
            picks.append(max(supported))
        elif totals:
            picks.append(max(totals))
        elif subtotals:
            picks.append(max(subtotals))
    return max(picks) if picks else None


def _aggregate_snapshot(
    rows: list[dict],
    documents: list[DocumentMeta],
    contract_anchor_ids: Optional[set[str]] = None,
) -> ReconciliationSnapshot:
    doc_type_by_id = {d.doc_id: d.doc_type for d in documents}
    contract_anchor_ids = contract_anchor_ids or set()
    all_debits_sum = 0.0
    bank_unattributed = 0
    contract_total: Optional[float] = None
    invoice_total: Optional[float] = None
    bank_line_keys: set[tuple] = set()
    bank_sum = 0.0
    bank_any = False
    bank_deduped = False

    contract_candidates: list[float] = []
    # Per-document invoice figures for the picker.
    invoice_by_doc: dict[str, dict[str, list[float]]] = {}
    milestone_sums: list[float] = []

    stated_total_debits: dict[str, float] = {}
    bank_unknown_skipped = 0

    for row in rows:
        dt = doc_type_by_id.get(row.get("doc_id"), "unknown")
        role = (row.get("amount_role") or "unknown").strip().lower()
        val = _parse_row_amount(row)
        if val is None or val < 1:
            continue

        # Cumulative account state / pre-aggregated totals are never transaction values.
        if role in NON_SUMMABLE_ROLES and role not in ("statement_total_debits",):
            continue

        if dt == "contract":
            if role == "total_contract_value":
                contract_candidates.append(val)
            elif role == "milestone_scheduled":
                milestone_sums.append(val)
        elif dt == "invoice":
            # invoice_referenced_contract_value is a claim about the contract, not a total.
            bucket = invoice_by_doc.setdefault(
                str(row.get("doc_id") or ""), {"total": [], "subtotal": [], "vat": []}
            )
            if role == "total_invoice":
                bucket["total"].append(val)
            elif role == "invoice_subtotal":
                bucket["subtotal"].append(val)
            elif role == "vat_tax":
                bucket["vat"].append(val)
        # payment_certificate is deliberately not aggregated: certified ≠ invoiced.
        elif dt == "bank_statement":
            doc_id = str(row.get("doc_id") or "")
            if role == "statement_total_debits":
                # Validates the summed rows; never added to them.
                stated_total_debits[doc_id] = max(stated_total_debits.get(doc_id, 0.0), val)
                continue
            if role not in BANK_OUTFLOW_ROLES:
                # Only positively-identified outflows count.
                if role == "unknown":
                    bank_unknown_skipped += 1
                continue
            page = int(row.get("source_page") or 0)
            txn_ref = str(row.get("txn_ref") or "").strip().lower()
            txn_date = str(row.get("txn_date") or "").strip().lower()
            # Dedupe on (ref, date, amount) — never payee; page only when no ref/date.
            if txn_ref or txn_date:
                bkey: tuple = ("txn", doc_id, txn_ref, txn_date, _bank_round_key(val))
            else:
                bkey = ("line", doc_id, page, _bank_round_key(val))
            if bkey in bank_line_keys:
                bank_deduped = True
                continue
            bank_line_keys.add(bkey)
            all_debits_sum += val
            # Only debits linked to the contract's anchors count as paid on it.
            anchors = set(row.get("anchor_ids") or [])
            if not contract_anchor_ids:
                # No contract in this document set — nothing to scope to.
                bank_sum += val
                bank_any = True
            elif anchors & contract_anchor_ids:
                bank_sum += val
                bank_any = True
            else:
                bank_unattributed += 1

    if contract_candidates:
        contract_total = max(contract_candidates)
    elif milestone_sums:
        contract_total = sum(milestone_sums)

    invoice_total = _pick_invoice_total(invoice_by_doc)

    bank_paid = bank_sum if bank_any else None

    variance: Optional[float] = None
    notes_parts: list[str] = []
    if bank_deduped:
        notes_parts.append("Bank total deduplicated (duplicate amount entities in graph).")

    # Completeness gate: extracted debits must reconcile with stated Total Debits,
    # else the bank total is withheld rather than shown wrong.
    bank_extraction_incomplete = False
    bank_debits_verified: Optional[float] = None
    bank_debits_stated: Optional[float] = None
    if stated_total_debits and all_debits_sum > 0:
        stated = sum(stated_total_debits.values())
        if stated > 0:
            drift = abs(all_debits_sum - stated)
            tolerance = max(1.0, stated * 0.005)  # 0.5% or 1 unit, whichever is larger
            if drift > tolerance:
                logger.warning(
                    "Bank debit validation FAILED: extracted debits total %.2f vs statement's "
                    "stated Total Debits %.2f (drift %.2f) — withholding bank total",
                    all_debits_sum, stated, drift,
                )
                bank_extraction_incomplete = True
                bank_debits_verified = round(all_debits_sum, 2)
                bank_debits_stated = round(stated, 2)
                bank_paid = None  # do not surface a number we could not verify
                notes_parts.append(
                    f"⚠ Extraction incomplete — {all_debits_sum:,.0f} of {stated:,.0f} EGP "
                    f"in bank debits verified. Bank total withheld until the remaining rows "
                    f"are extracted."
                )
            else:
                logger.info(
                    "Bank debit validation passed: extracted %.2f vs stated %.2f",
                    all_debits_sum, stated,
                )
    if bank_any and bank_unattributed:
        logger.info(
            "Bank aggregation: %d debit(s) excluded as unrelated to this contract "
            "(contract-attributed total %.2f of %.2f total debits)",
            bank_unattributed, bank_sum, all_debits_sum,
        )
    if bank_unknown_skipped:
        logger.info(
            "Bank aggregation skipped %d amount(s) with role='unknown' (not summed by design)",
            bank_unknown_skipped,
        )
    if bank_paid is None and bank_unknown_skipped:
        notes_parts.append(
            f"Bank total unavailable: {bank_unknown_skipped} bank amount(s) could not be "
            "classified as debit or credit during extraction."
        )
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
        bank_extraction_incomplete=bank_extraction_incomplete,
        bank_debits_verified=bank_debits_verified,
        bank_debits_stated=bank_debits_stated,
    )


# Keyword nearest BEFORE an amount in finding prose decides its field.
_ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "contract_total": ("contract", "agreement", "عقد", "اتفاق"),
    "invoice_total": ("invoice", "billed", "فاتور"),
    "bank_paid_total": (
        "bank paid", "total paid", "bank", "paid", "transfer",
        "disburse", "remitt", "دفع", "سداد", "تحويل",
    ),
}
_LABEL_WINDOW = 50  # chars before an amount to search for a role keyword


def _role_for_window(window: str) -> Optional[str]:
    """Return the role whose keyword sits closest to the end of `window` (nearest the amount)."""
    w = window.lower()
    best_role: Optional[str] = None
    best_pos = -1
    for role, keywords in _ROLE_KEYWORDS.items():
        for kw in keywords:
            idx = w.rfind(kw)
            if idx > best_pos:
                best_pos = idx
                best_role = role
    return best_role


def _label_grounded_amounts(findings: list[Finding]) -> dict[str, float]:
    """Field → largest amount from critical findings, only when a role keyword precedes it."""
    out: dict[str, float] = {}
    for f in findings:
        if f.severity != "critical":
            continue
        text = f"{f.title}. {f.description or ''}"
        for am in extract_amounts(text):
            val = am.get("value")
            pos = int(am.get("position", 0) or 0)
            if val is None or val < _MIN_HEADLINE:
                continue
            role = _role_for_window(text[max(0, pos - _LABEL_WINDOW):pos])
            if role is None:
                continue
            if role not in out or val > out[role]:
                out[role] = float(val)
    return out


def _fill_snapshot_gaps(
    snapshot: ReconciliationSnapshot,
    inferred: dict[str, float],
) -> ReconciliationSnapshot:
    """Fill only empty snapshot fields; structured values are never overwritten."""
    fields = ("contract_total", "invoice_total", "bank_paid_total")
    filled: dict[str, float] = {
        field: inferred[field]
        for field in fields
        if getattr(snapshot, field) is None and field in inferred
        # Never re-introduce a withheld bank total.
        and not (field == "bank_paid_total" and snapshot.bank_extraction_incomplete)
    }
    if not filled:
        return snapshot

    update: dict = dict(filled)
    ct = filled.get("contract_total", snapshot.contract_total)
    bp = filled.get("bank_paid_total", snapshot.bank_paid_total)
    if ct is not None and bp is not None:
        update["variance_vs_contract"] = round(bp - ct, 2)

    label = ", ".join(field.replace("_total", "").replace("_", " ") for field in filled)
    note = f"Inferred from labeled finding text: {label}."
    update["notes"] = " ".join(p for p in (snapshot.notes, note) if p)
    return snapshot.model_copy(update=update)


def _contradictions_to_rows(contradictions: list[dict]) -> list[EntityConflictRow]:
    rows: list[EntityConflictRow] = []
    seen: set[tuple] = set()
    for c in contradictions:
        d1 = c.get("doc1_id") or ""
        d2 = c.get("doc2_id") or ""
        v1 = str(c.get("value1") or c.get("norm1") or "").strip()
        v2 = str(c.get("value2") or c.get("norm2") or "").strip()
        if not d1 or not d2 or d1 == d2:
            continue
        # Dedupe on numeric pair + reason, not document pair.
        p1 = parse_monetary_amount(v1)
        p2 = parse_monetary_amount(v2)
        num_key = frozenset({
            round(p1, 2) if p1 is not None else v1.lower(),
            round(p2, 2) if p2 is not None else v2.lower(),
        })
        key = (num_key, str(c.get("comparison_reason") or ""))
        if key in seen:
            continue
        seen.add(key)
        anchor = str(c.get("anchor_value") or "").strip() or None
        # Reason over bare anchor — it says why.
        reason = str(c.get("comparison_reason") or "").strip()
        label = reason or anchor or "Cross-document amount"
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
                conflict_type=reason[:200] if reason else "amount",
                anchor_hint=anchor,
            )
        )
    return rows[:30]


def _row_numeric_key(v1: str, v2: str) -> frozenset:
    """Order-independent identity for a conflict row, keyed on its two amounts."""
    p1 = parse_monetary_amount(v1)
    p2 = parse_monetary_amount(v2)
    return frozenset({
        round(p1, 2) if p1 is not None else v1.strip().lower(),
        round(p2, 2) if p2 is not None else v2.strip().lower(),
    })


def _findings_to_conflict_rows(findings: list[Finding]) -> list[EntityConflictRow]:
    """Conflict rows from the same findings the Findings tab shows (≥2 headline amounts)."""
    rows: list[EntityConflictRow] = []
    for f in findings:
        if f.severity not in ("critical", "warning"):
            continue
        text = f"{f.title}. {f.description or ''}"
        amounts: list[float] = []
        for am in extract_amounts(text):
            val = am.get("value")
            if val is None or float(val) < _MIN_HEADLINE:
                continue
            fv = round(float(val), 2)
            if fv not in amounts:
                amounts.append(fv)
        if len(amounts) < 2:
            continue
        amounts.sort(reverse=True)
        a, b = amounts[0], amounts[1]
        rows.append(
            EntityConflictRow(
                entity_label=f.title[:200],
                doc_a_id=f.source_doc_id or "",
                doc_a_value=f"{a:,.2f}",
                doc_b_id=f.conflicting_doc_id or "",
                doc_b_value=f"{b:,.2f}",
                severity=f.severity,  # already narrowed to critical|warning above
                conflict_type="amount finding",
                anchor_hint=None,
            )
        )
    return rows


def _merge_conflict_rows(
    finding_rows: list[EntityConflictRow],
    graph_rows: list[EntityConflictRow],
) -> list[EntityConflictRow]:
    """Findings-derived rows first; add graph rows only for amount pairs not already shown."""
    out: list[EntityConflictRow] = list(finding_rows)
    seen: set[frozenset] = {_row_numeric_key(r.doc_a_value, r.doc_b_value) for r in out}
    for r in graph_rows:
        k = _row_numeric_key(r.doc_a_value, r.doc_b_value)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out[:30]


def _snapshot_has_numbers(s: ReconciliationSnapshot) -> bool:
    return any(
        x is not None for x in (s.contract_total, s.invoice_total, s.bank_paid_total)
    ) or s.bank_extraction_incomplete


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
    """Reconciliation snapshot + conflict rows from graph amounts, findings as fallback."""
    documents = _normalize_documents(documents)
    doc_ids = [d.doc_id for d in documents]
    amount_rows = _fetch_amount_rows(doc_ids)
    snapshot = _aggregate_snapshot(amount_rows, documents, _contract_anchor_ids(doc_ids))

    # Gap-fill only empty fields from label-grounded finding amounts.
    inferred = _label_grounded_amounts(findings) if findings else {}
    snapshot = _fill_snapshot_gaps(snapshot, inferred)

    if not _snapshot_has_numbers(snapshot):
        snapshot = None

    # Same findings as the Findings tab, plus uncovered graph-only pairs.
    finding_rows = _findings_to_conflict_rows(findings or [])
    graph_rows = _contradictions_to_rows(find_contradictions(doc_ids))
    conflicts = _merge_conflict_rows(finding_rows, graph_rows)
    return snapshot, conflicts
