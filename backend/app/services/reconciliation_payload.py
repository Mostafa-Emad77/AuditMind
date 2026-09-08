"""Build reconciliation snapshot and entity conflict rows for AuditReport."""
from __future__ import annotations

import logging
from collections import Counter
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
from app.services.graph_builder import (
    audit_identifier_values,
    find_contradictions,
    get_driver,
)
from app.services.vector_store import get_all_chunks_for_docs
from app.utils.canonical_id import normalize_identifier
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
                       e.amount_currency AS amount_currency,
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


def _refs_named_alongside_an_audit_identifier(
    doc_ids: list[str], known_refs: set[str], identifier_cores: set[str]
) -> set[str]:
    """Transaction refs whose own row text also names a contract/invoice of this audit.

    Rows survive ingestion intact now, so a payment can be tied to the contract it
    names in its own description — no LLM-emitted edge required.
    """
    if not doc_ids or not known_refs or not identifier_cores:
        return set()
    try:
        chunks = get_all_chunks_for_docs(doc_ids)
    except Exception as e:
        logger.warning("Could not read rows for payment attribution: %s", e)
        return set()

    ref_cores = {ref: normalize_identifier(ref) for ref in known_refs if ref}
    linked: set[str] = set()
    for chunk in chunks:
        for line in (chunk.get("text") or "").splitlines():
            core = normalize_identifier(line)
            if not core or not any(idc in core for idc in identifier_cores):
                continue
            for ref, ref_core in ref_cores.items():
                if ref_core and ref_core in core:
                    linked.add(ref)
    if linked:
        logger.info("Payment attribution: %d row(s) name an audited identifier", len(linked))
    return linked


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


# Arithmetic a document states about itself. Only exact identities are checked, so a
# warning means figures were missed or mis-tagged, not that the document is unusual.
_ARITHMETIC_TOLERANCE = 1.0


def _document_integrity_warnings(
    rows: list[dict], documents: list[DocumentMeta]
) -> list[str]:
    """Check each document's own arithmetic against what extraction actually found.

    The bank statement has its stated Total Debits; these are the equivalents for the
    other document types, so a silent extraction miss is visible there too.
    """
    by_doc: dict[str, dict[str, list[float]]] = {}
    names = {d.doc_id: d.filename for d in documents}
    types = {d.doc_id: d.doc_type for d in documents}
    for row in rows:
        val = _parse_row_amount(row)
        doc_id = str(row.get("doc_id") or "")
        if val is None or not doc_id:
            continue
        role = (row.get("amount_role") or "unknown").strip().lower()
        by_doc.setdefault(doc_id, {}).setdefault(role, []).append(val)

    warnings: list[str] = []
    for doc_id, roles in by_doc.items():
        name = names.get(doc_id, doc_id[:8])
        doc_type = types.get(doc_id, "unknown")

        def _one(role: str) -> Optional[float]:
            values = roles.get(role) or []
            return max(values) if values else None

        if doc_type == "invoice":
            total, subtotal = _one("total_invoice"), _one("invoice_subtotal")
            vat = _one("vat_tax")
            # A late fee is a real component of the total, not a discrepancy.
            fee = _one("late_fee") or 0.0
            if total is not None and subtotal is not None and vat is not None:
                expected = subtotal + vat + fee
                if abs(total - expected) > _ARITHMETIC_TOLERANCE:
                    parts = f"subtotal {subtotal:,.2f} + VAT {vat:,.2f}"
                    if fee:
                        parts += f" + late fee {fee:,.2f}"
                    warnings.append(
                        f"{name}: {parts} = {expected:,.2f}, "
                        f"but the stated total is {total:,.2f}."
                    )

        if doc_type == "balance_sheet":
            assets, liabilities = _one("total_assets"), _one("total_liabilities")
            equity = _one("total_equity")
            if assets is not None and liabilities is not None and equity is not None:
                if abs(assets - (liabilities + equity)) > _ARITHMETIC_TOLERANCE:
                    warnings.append(
                        f"{name}: liabilities {liabilities:,.2f} + equity {equity:,.2f} = "
                        f"{liabilities + equity:,.2f}, but total assets are {assets:,.2f}."
                    )

        if doc_type == "contract":
            total = _one("total_contract_value")
            milestones = roles.get("milestone_scheduled") or []
            scheduled = sum(milestones)
            # Only a shortfall is reported: it means schedule lines were missed. An
            # overshoot usually means a figure was double-counted elsewhere and is
            # left to the contradiction checks.
            if total is not None and len(milestones) >= 2 and scheduled < total - _ARITHMETIC_TOLERANCE:
                warnings.append(
                    f"{name}: {len(milestones)} scheduled milestone(s) sum to "
                    f"{scheduled:,.2f} of a stated contract total of {total:,.2f} — "
                    f"schedule lines may be missing."
                )

    for warning in warnings:
        logger.warning("Document arithmetic check: %s", warning)
    return warnings


def _snapshot_currency(rows: list[dict]) -> tuple[str, set[str]]:
    """The currency these figures are in, plus any others present.

    Everything was previously labelled EGP regardless of the documents, so a USD or
    SAR audit rendered every card with the wrong currency. Totals across different
    currencies are not comparable, so the caller warns when more than one appears.
    """
    seen = Counter(
        code
        for code in (str(r.get("amount_currency") or "").strip().upper() for r in rows)
        if code
    )
    if not seen:
        return "EGP", set()
    primary, _ = seen.most_common(1)[0]
    return primary, set(seen) - {primary}


def _aggregate_snapshot(
    rows: list[dict],
    documents: list[DocumentMeta],
    contract_anchor_ids: Optional[set[str]] = None,
    contract_linked_refs: Optional[set[str]] = None,
) -> ReconciliationSnapshot:
    doc_type_by_id = {d.doc_id: d.doc_type for d in documents}
    contract_anchor_ids = contract_anchor_ids or set()
    # Debit refs are stored lowercased; match case-insensitively.
    contract_linked_refs = {r.strip().lower() for r in (contract_linked_refs or set()) if r}
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
    bank_debits: list[dict] = []

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
            bank_debits.append({
                "doc_id": doc_id,
                "page": int(row.get("source_page") or 0),
                "ref": str(row.get("txn_ref") or "").strip().lower(),
                "date": str(row.get("txn_date") or "").strip().lower(),
                "value": val,
                "anchors": set(row.get("anchor_ids") or []),
            })

    # A statement's own recap ("Bank total paid: EGP 226,700") carries no transaction
    # identity, and counting it alongside the rows it summarises double-counts them.
    # When any row is identified, unidentified amounts are recaps, not transactions.
    identified = [d for d in bank_debits if d["ref"] or d["date"]]
    if identified and len(identified) < len(bank_debits):
        logger.info(
            "Bank aggregation: ignoring %d debit amount(s) with no transaction identity "
            "(summary recaps, not rows)", len(bank_debits) - len(identified),
        )
        bank_debits = identified

    for debit in bank_debits:
        # One transaction per (ref, amount): the same row recapped on a later page
        # repeats its ref, so keying on the ref alone collapses the duplicate.
        if debit["ref"]:
            bkey: tuple = ("ref", debit["doc_id"], debit["ref"], _bank_round_key(debit["value"]))
        elif debit["date"]:
            bkey = ("date", debit["doc_id"], debit["date"], _bank_round_key(debit["value"]))
        else:
            bkey = ("line", debit["doc_id"], debit["page"], _bank_round_key(debit["value"]))
        if bkey in bank_line_keys:
            bank_deduped = True
            continue
        bank_line_keys.add(bkey)
        all_debits_sum += debit["value"]
        # A debit is paid on this contract when a graph edge links it to the contract's
        # anchors, or when its own row text names one of the audit's identifiers.
        attributed = bool(debit["anchors"] & contract_anchor_ids) or (
            debit["ref"] in contract_linked_refs
        )
        if not contract_anchor_ids or attributed:
            bank_sum += debit["value"]
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

    integrity = _document_integrity_warnings(rows, documents)
    currency, other_currencies = _snapshot_currency(rows)
    if other_currencies:
        warning = (
            f"Amounts appear in more than one currency ({currency}, "
            f"{', '.join(sorted(other_currencies))}); totals mix them and are not comparable."
        )
        logger.warning("Reconciliation: %s", warning)
        integrity.append(warning)

    return ReconciliationSnapshot(
        currency=currency,
        contract_total=round(contract_total, 2) if contract_total is not None else None,
        invoice_total=round(invoice_total, 2) if invoice_total is not None else None,
        bank_paid_total=round(bank_paid, 2) if bank_paid is not None else None,
        variance_vs_contract=variance,
        notes=" ".join(notes_parts) if notes_parts else None,
        bank_extraction_incomplete=bank_extraction_incomplete,
        bank_debits_verified=bank_debits_verified,
        bank_debits_stated=bank_debits_stated,
        integrity_warnings=integrity,
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


# Fixed order so a row always reads contract → invoice → bank.
_DOC_TYPE_ORDER: tuple[str, ...] = ("contract", "invoice", "payment_certificate", "bank_statement")


def _amount_variants(v: float) -> tuple[str, ...]:
    """Surface forms an amount may take in evidence text ("226,700.00", "226,700", …)."""
    return (f"{v:,.2f}", f"{v:,.0f}", f"{v:.2f}", f"{v:.0f}")


def _headline_amounts(text: str) -> list[float]:
    """Distinct headline amounts in order of appearance."""
    out: list[float] = []
    for am in extract_amounts(text):
        val = am.get("value")
        if val is None or float(val) < _MIN_HEADLINE:
            continue
        fv = round(float(val), 2)
        if fv not in out:
            out.append(fv)
    return out


def _amount_doc_index(amount_rows: list[dict]) -> dict[float, set[str]]:
    """value → documents that actually contain it, straight from the graph."""
    idx: dict[float, set[str]] = {}
    for row in amount_rows:
        val = _parse_row_amount(row)
        doc_id = str(row.get("doc_id") or "")
        if val is None or not doc_id:
            continue
        idx.setdefault(round(val, 2), set()).add(doc_id)
    return idx


def _ground_amounts(
    finding: Finding, documents: list[DocumentMeta], amount_index: dict[float, set[str]]
) -> dict[float, str]:
    """Map each headline amount in a finding to the document it came from.

    Attribution is established, never guessed: an amount quoted verbatim in an
    evidence line belongs to that line's document, and otherwise an amount that was
    extracted from exactly one document belongs to that one. 
    """
    text = f"{finding.title}. {finding.description or ''}"
    by_name = {d.filename: d.doc_id for d in documents}
    grounded: dict[float, str] = {}
    for amt in _headline_amounts(text):
        variants = _amount_variants(amt)
        doc_id = next(
            (
                did
                for line in (finding.evidence or [])
                for name, did in by_name.items()
                if line.startswith(name) and any(v in line for v in variants)
            ),
            "",
        )
        if not doc_id:
            owners = amount_index.get(amt, set())
            if len(owners) == 1:
                doc_id = next(iter(owners))
        if doc_id:
            grounded[amt] = doc_id
    return grounded


def _findings_to_conflict_rows(
    findings: list[Finding],
    documents: list[DocumentMeta],
    amount_rows: list[dict],
) -> list[EntityConflictRow]:
    """Conflict rows from the same findings the Findings tab shows.
    Each headline amount in a finding is mapped to the document it came from, and
    then the two largest amounts from different documents are paired into a row.
    """
    type_by_id = {d.doc_id: d.doc_type for d in documents}
    amount_index = _amount_doc_index(amount_rows)

    def _rank(doc_id: str) -> int:
        dt = type_by_id.get(doc_id, "")
        return _DOC_TYPE_ORDER.index(dt) if dt in _DOC_TYPE_ORDER else len(_DOC_TYPE_ORDER)

    rows: list[EntityConflictRow] = []
    for f in findings:
        if f.severity not in ("critical", "warning"):
            continue
        grounded = _ground_amounts(f, documents, amount_index)
        pairs = sorted(grounded.items(), key=lambda kv: (_rank(kv[1]), -kv[0]))
        first = pairs[0] if pairs else None
        second = next((kv for kv in pairs[1:] if kv[1] != first[1]), None) if first else None
        if first is None or second is None:
            continue
        (amt_a, doc_a), (amt_b, doc_b) = first, second
        label_a = (type_by_id.get(doc_a) or "document").replace("_", " ")
        label_b = (type_by_id.get(doc_b) or "document").replace("_", " ")
        rows.append(
            EntityConflictRow(
                entity_label=f.title[:200],
                doc_a_id=doc_a,
                doc_a_value=f"{amt_a:,.2f}",
                doc_b_id=doc_b,
                doc_b_value=f"{amt_b:,.2f}",
                severity=f.severity,  # already narrowed to critical|warning above
                conflict_type=f"{label_a} vs {label_b} (from finding)",
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
    ) or s.bank_extraction_incomplete or bool(s.integrity_warnings)


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
    bank_doc_ids = [d.doc_id for d in documents if d.doc_type == "bank_statement"]
    known_refs = {
        str(r.get("txn_ref") or "").strip()
        for r in amount_rows
        if r.get("txn_ref") and str(r.get("doc_id")) in bank_doc_ids
    }
    linked_refs = _refs_named_alongside_an_audit_identifier(
        bank_doc_ids, known_refs, audit_identifier_values(doc_ids)
    )
    snapshot = _aggregate_snapshot(
        amount_rows, documents, _contract_anchor_ids(doc_ids), linked_refs
    )

    # Gap-fill only empty fields from label-grounded finding amounts.
    inferred = _label_grounded_amounts(findings) if findings else {}
    snapshot = _fill_snapshot_gaps(snapshot, inferred)

    if not _snapshot_has_numbers(snapshot):
        snapshot = None

    # Same findings as the Findings tab, plus uncovered graph-only pairs.
    finding_rows = _findings_to_conflict_rows(findings or [], documents, amount_rows)
    graph_rows = _contradictions_to_rows(find_contradictions(doc_ids))
    conflicts = _merge_conflict_rows(finding_rows, graph_rows)
    return snapshot, conflicts
