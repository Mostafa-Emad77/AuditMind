"""LLM calls that assess/adjudicate candidate cross-document contradictions."""
import json
import logging
from typing import Any

from app.models.schemas import Finding, ChecklistItem
from app.utils.llm_factory import get_llm
from app.utils.llm_json import extract_json_obj as _extract_json_obj, normalize_llm_content

logger = logging.getLogger(__name__)


async def _llm_assess_amount_pair(
    value1: str,
    value2: str,
    context: str,
    evidence_lines: list[str],
) -> dict | None:
    """
    Ask LLM if a pair is a meaningful contradiction (like-with-like only).
    Returns structured decision dict or None on failure.
    """
    llm = get_llm(temperature=0.0, role="relation")
    prompt = (
        "You are a forensic audit checker. Decide if two numeric values are a REAL contradiction.\n"
        "You MUST reject apples-to-oranges comparisons.\n\n"
        "REFERENCE NUMBERS: Only cite reference numbers, contract IDs, or document numbers that\n"
        "appear as literal visible text in the evidence below. Never reference, compare against,\n"
        "or flag as 'missing' an identifier that is not directly quoted from a source document.\n"
        "Internal system identifiers (UUIDs, session IDs, audit IDs) must never appear in your output.\n\n"
        "Reject comparison if any of these apply:\n"
        "- one value is opening/closing balance and the other is invoice/contract total\n"
        "- one value is a single payment installment and the other is a full total\n"
        "- one value looks like an article/section/year/reference number\n"
        "- context evidence does not show they refer to the same business fact\n"
        "- one amount is an explicitly scheduled contract advance/down payment and the other is a "
        "different schedule line (milestone/retainer total) — not a contradiction unless the "
        "contract itself is inconsistent\n\n"
        "Input:\n"
        f"value1: {value1}\n"
        f"value2: {value2}\n"
        f"context: {context}\n"
        f"evidence: {json.dumps(evidence_lines, ensure_ascii=False)}\n\n"
        "Return JSON only with this exact schema:\n"
        "{\n"
        "  \"is_valid_comparison\": true/false,\n"
        "  \"is_contradiction\": true/false,\n"
        "  \"severity\": \"critical|warning|ok\",\n"
        "  \"confidence\": 0.0-1.0,\n"
        "  \"explanation\": \"short explanation\"\n"
        "}\n"
    )
    try:
        resp = await llm.ainvoke(prompt)
        content = normalize_llm_content(resp)
        data = _extract_json_obj(content)
        if not data:
            return None
        out = {
            "is_valid_comparison": bool(data.get("is_valid_comparison", False)),
            "is_contradiction": bool(data.get("is_contradiction", False)),
            "severity": str(data.get("severity", "ok")).lower(),
            "confidence": float(data.get("confidence", 0.7)),
            "explanation": str(data.get("explanation", "")),
        }
        if out["severity"] not in ("critical", "warning", "ok"):
            out["severity"] = "warning" if out["is_contradiction"] else "ok"
        out["confidence"] = max(0.0, min(1.0, out["confidence"]))
        return out
    except Exception as e:
        logger.debug("LLM pair assessment failed: %s", e)
        return None


async def _llm_adjudicate_check(
    item: ChecklistItem,
    results: list[dict],
    documents: list[Any],
    *,
    max_findings: int = 4,
) -> list[Finding]:
    """
    Let LLM review retrieved snippets and emit only meaningful findings.
    Used as fallback instead of brittle first-number comparisons.
    """
    if not results:
        return []
    by_doc = {d.doc_id: d for d in documents}
    snippets: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for r in results:
        doc_id = r.get("doc_id", "")
        text = (r.get("text") or r.get("value") or "").strip()
        if not text:
            continue
        key = (doc_id, text[:180])
        if key in seen:
            continue
        seen.add(key)
        snippets.append({
            "doc_id": doc_id,
            "doc_name": getattr(by_doc.get(doc_id), "filename", f"document {len(snippets) + 1}"),
            "doc_type": (getattr(by_doc.get(doc_id), "doc_type", "") or "unknown"),
            "page": int(r.get("page", 0) or 0),
            "text": text[:900],
        })
        if len(snippets) >= 14:
            break
    if len(snippets) < 2:
        return []

    # Raw doc_id UUIDs never reach the prompt (the model echoes them as fake refs).
    prompt_snippets = [
        {k: v for k, v in s.items() if k != "doc_id"} for s in snippets
    ]

    llm = get_llm(temperature=0.0, role="relation")
    prompt = (
        "You are a strict financial audit reviewer.\n"
        "Analyze the checklist check against document snippets and output ONLY real findings.\n"
        "Do NOT compare apples-to-oranges (partial payment vs full total, opening balance vs invoice total, etc).\n"
        "REFERENCE NUMBERS: Only cite reference numbers, contract IDs, or document numbers that appear as\n"
        "literal visible text within the provided snippets. Never reference, compare against, or flag as\n"
        "'missing' any identifier that is not directly quoted from a snippet. Internal system metadata\n"
        "(UUIDs, session IDs, audit IDs) must never appear in a finding's title, description, or recommendation.\n"
        "Prefer like-with-like reconciliations: total paid vs contract total vs invoice total; schedule amounts vs milestone payments.\n\n"
        "ADVANCE PAYMENT — avoid false-positive warnings:\n"
        "- If snippets show the contract payment schedule explicitly includes an advance payment "
        "(e.g. \"Advance Payment\", \"دفعة مقدمة\", down payment) for amount A on a date, and the bank "
        "shows an outgoing payment matching A (same order of magnitude and timing), treat that advance "
        "as AUTHORIZED by the contract as its own line — not as a mistake.\n"
        "- Do NOT emit warnings that the advance is \"not aligned with standard terms\", "
        "\"not linked to invoice milestones\", \"should reconcile to later milestones\", or "
        "\"fragmented vs invoice line items\" when the contract clearly lists the advance separately. "
        "Advances are often absent from recurring invoices by design.\n"
        "- Only flag an advance if evidence shows there is NO contractual line for that payment, "
        "or the amount/date clearly conflicts with the written schedule.\n\n"
        "CONSOLIDATION:\n"
        "- If several issues are the same root cause (e.g. total bank paid vs contract total overpayment), "
        "output a single clearest critical finding instead of multiple near-duplicate titles.\n\n"
        f"Checklist item: {item.description}\n"
        f"Check type: {item.check_type}\n"
        f"Snippets (indexed): {json.dumps(prompt_snippets, ensure_ascii=False)}\n\n"
        "Return JSON only:\n"
        "{\n"
        "  \"findings\": [\n"
        "    {\n"
        "      \"severity\": \"critical|warning|ok\",\n"
        "      \"title\": \"short title\",\n"
        "      \"description\": \"why this is a finding\",\n"
        "      \"confidence\": 0.0-1.0,\n"
        "      \"evidence_indices\": [0,1],\n"
        "      \"recommendation\": \"short recommendation\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "If no reliable finding, return {\"findings\": []}."
    )
    try:
        resp = await llm.ainvoke(prompt)
        content = normalize_llm_content(resp)
        data = _extract_json_obj(content) or {"findings": []}
        raw_findings = data.get("findings", [])
        if not isinstance(raw_findings, list):
            return []
        out: list[Finding] = []
        for f in raw_findings[:max_findings]:
            if not isinstance(f, dict):
                continue
            sev = str(f.get("severity", "warning")).lower()
            if sev not in ("critical", "warning", "ok"):
                sev = "warning"
            conf = float(f.get("confidence", 0.75))
            conf = max(0.0, min(1.0, conf))
            idxs = f.get("evidence_indices", [])
            evidence: list[str] = []
            ev_doc_ids: list[str] = []
            if isinstance(idxs, list):
                for idx in idxs[:4]:
                    if isinstance(idx, int) and 0 <= idx < len(snippets):
                        s = snippets[idx]
                        evidence.append(
                            f"{s['doc_name']}, page {s['page']}: {s['text'][:220]}"
                        )
                        ev_doc_ids.append(s["doc_id"])
            if not evidence:
                for s in snippets[:2]:
                    evidence.append(f"{s['doc_name']}, page {s['page']}: {s['text'][:220]}")
                    ev_doc_ids.append(s["doc_id"])
            source_doc_id = ev_doc_ids[0] if ev_doc_ids else None
            conflicting_doc_id = next((d for d in ev_doc_ids[1:] if d != source_doc_id), None)
            title = str(f.get("title", "")).strip() or f"LLM Review: {item.description[:80]}"
            description = str(f.get("description", "")).strip()
            if not description:
                continue
            out.append(Finding(
                checklist_item_id=item.item_id,
                severity=sev,
                title=title,
                description=description,
                confidence_score=conf,
                source_doc_id=source_doc_id,
                conflicting_doc_id=conflicting_doc_id,
                evidence=evidence,
                recommendation=str(f.get("recommendation", "")).strip() or "Review source records and reconcile supporting evidence.",
            ))
        return out
    except Exception as e:
        logger.debug("LLM checklist adjudication failed: %s", e)
        return []
