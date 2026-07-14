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


# Role keywords used to attach a loose amount in finding prose to a reconciliation
# field. We pick the keyword occurring CLOSEST before the amount (not by rank), so
# "bank paid 226,700 vs contract 200,000" maps each number to the right field.
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
    """
    Map reconciliation fields → amount using only LABEL-GROUNDED evidence from
    critical findings. An amount is assigned to a field only when a role keyword
    appears just before it in the text; amounts with no nearby label are ignored
    (no positional / sort-order guessing). Returns the largest amount per field.
    """
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
    """
    Fill ONLY the empty fields of a structured snapshot with label-grounded
    inferences. Structured (role-tagged) values are authoritative and are never
    overwritten. Variance is recomputed when both sides become known.
    """
    fields = ("contract_total", "invoice_total", "bank_paid_total")
    filled: dict[str, float] = {
        field: inferred[field]
        for field in fields
        if getattr(snapshot, field) is None and field in inferred
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

    # Gap-fill only the fields the structured path left empty, using label-grounded
    # amounts from critical findings. Structured values are never overwritten.
    inferred = _label_grounded_amounts(findings) if findings else {}
    snapshot = _fill_snapshot_gaps(snapshot, inferred)

    if not _snapshot_has_numbers(snapshot):
        snapshot = None

    conflicts = _contradictions_to_rows(find_contradictions(doc_ids))
    return snapshot, conflicts
