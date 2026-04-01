"""Cross-Checker Agent — the core contradiction detection engine."""
import json
import logging
import re
from typing import Any
from langgraph.config import get_stream_writer
from langchain_core.messages import AIMessage

from app.models.state import AuditState
from app.models.schemas import ReasoningStep, Finding, ChecklistItem
from app.tools.hybrid_retriever import search_hybrid_rag
from app.tools.neo4j_tools import detect_graph_contradictions
from app.tools.web_search import search_web
from app.utils.arabic_normalizer import extract_amounts
from app.utils.money_parse import parse_monetary_amount, amounts_within_tiny_tolerance
from app.services.vector_store import get_all_chunks_for_docs
from app.utils.llm_factory import get_llm
from app.utils.text_amount_scan import amounts_from_text_scan

logger = logging.getLogger(__name__)


def _emit(writer, step_type: str, content: str, **kwargs) -> ReasoningStep:
    step = ReasoningStep(agent="cross_checker", step_type=step_type, content=content, **kwargs)
    writer({"type": "reasoning_step", "step": step.model_dump(mode="json")})
    return step


def _extract_json_obj(text: str) -> dict | None:
    """Extract first JSON object from model output (brace-balanced scan, no regex)."""
    if not text:
        return None
    t = text.strip()
    try:
        parsed = json.loads(t)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    quote = ""
    i = start
    while i < len(t):
        c = t[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == quote:
                in_str = False
            i += 1
            continue
        if c in "\"'":
            in_str = True
            quote = c
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = t[start : i + 1]
                try:
                    parsed = json.loads(chunk)
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
        i += 1
    return None


def _llm_assess_amount_pair(
    value1: str,
    value2: str,
    context: str,
    evidence_lines: list[str],
) -> dict | None:
    """
    Ask LLM if a pair is a meaningful contradiction (like-with-like only).
    Returns structured decision dict or None on failure.
    """
    llm = get_llm(temperature=0.0)
    prompt = (
        "You are a forensic audit checker. Decide if two numeric values are a REAL contradiction.\n"
        "You MUST reject apples-to-oranges comparisons.\n\n"
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
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        data = _extract_json_obj(str(content))
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


def _llm_adjudicate_check(
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
            "doc_name": getattr(by_doc.get(doc_id), "filename", doc_id[:8]),
            "doc_type": (getattr(by_doc.get(doc_id), "doc_type", "") or "unknown"),
            "page": int(r.get("page", 0) or 0),
            "text": text[:900],
        })
        if len(snippets) >= 14:
            break
    if len(snippets) < 2:
        return []

    llm = get_llm(temperature=0.0)
    prompt = (
        "You are a strict financial audit reviewer.\n"
        "Analyze the checklist check against document snippets and output ONLY real findings.\n"
        "Do NOT compare apples-to-oranges (partial payment vs full total, opening balance vs invoice total, etc).\n"
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
        f"Snippets (indexed): {json.dumps(snippets, ensure_ascii=False)}\n\n"
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
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        data = _extract_json_obj(str(content)) or {"findings": []}
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


def _evidence_line_backs_value(evidence_line: str, monetary_raw: str) -> bool:
    """True if the evidence text contains a parsed amount matching monetary_raw."""
    target = parse_monetary_amount(str(monetary_raw).strip())
    if target is None:
        return False
    tail = evidence_line.split(":", 1)[-1].strip() if ":" in evidence_line else evidence_line
    direct = parse_monetary_amount(tail)
    if direct is not None and amounts_within_tiny_tolerance(direct, target):
        return True
    for am in extract_amounts(tail):
        v = am.get("value")
        if v is not None and amounts_within_tiny_tolerance(float(v), target):
            return True
    whole = parse_monetary_amount(evidence_line)
    return whole is not None and amounts_within_tiny_tolerance(whole, target)


def _amounts_in_snippet(text: str) -> list[float]:
    out: list[float] = []
    for am in extract_amounts(text):
        v = am.get("value")
        if v is not None:
            out.append(float(v))
            continue
        raw = (am.get("raw") or "").strip()
        p = parse_monetary_amount(raw)
        if p is not None:
            out.append(float(p))
    return out


def _evidence_snippets_back_values(evidence: list[str], *values: float) -> bool:
    """True if each value appears (within tiny tolerance) in at least one evidence snippet."""
    if not values:
        return True
    collected: list[float] = []
    for ev in evidence:
        tail = ev.split(":", 1)[-1].strip() if ":" in ev else ev
        collected.extend(_amounts_in_snippet(tail))
    for val in values:
        if not any(amounts_within_tiny_tolerance(c, val) for c in collected):
            return False
    return True


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


def _finding_numeric_signature(f: Finding) -> tuple[float, ...]:
    """Stable amount set from title + description only (evidence would add noisy balance columns)."""
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
            nums.append(round(p, -2))
    for p in amounts_from_text_scan(text, min_value=1000):
        if _amount_for_dedup_signature(p):
            nums.append(round(p, -2))
    return tuple(sorted(set(nums)))


def _finding_dedup_key(f: Finding) -> tuple:
    # Deprecated: returning None as this strict tuple equality is replaced by the custom loop in Phase 4.
    pass


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


def _graph_context_for_compare(c: dict, doc1: str, doc2: str) -> str:
    parts = [
        "Detected via knowledge graph: amounts linked through a shared anchor entity.",
        f"Documents: {doc1} (page {c.get('page1', 0)}) and {doc2} (page {c.get('page2', 0)}).",
    ]
    if c.get("anchor_type") or c.get("anchor_value"):
        parts.append(
            f"Anchor: {c.get('anchor_type', 'entity')} = {c.get('anchor_value', '')!r} "
            f"(id={c.get('anchor_entity_id', '')})."
        )
    return " ".join(parts)


def _is_bank_doc(doc: Any) -> bool:
    fn = (getattr(doc, "filename", "") or "").lower()
    dt = (getattr(doc, "doc_type", "") or "").lower()
    return dt == "bank_statement" or any(k in fn for k in ("bank", "statement", "stmt"))


def _classify_check_intents(description: str, check_type: str) -> set[str]:
    """
    LLM-based checklist intent classification.
    Returns a set of intents among:
      schedule, subtotal, bank_contract_recon, invoice_bank_recon, milestone_bank_recon, generic
    """
    prompt = (
        "Classify this audit checklist item into one or more intents.\n"
        "Allowed intents: schedule, subtotal, bank_contract_recon, invoice_bank_recon, milestone_bank_recon, generic.\n"
        "Return JSON only:\n"
        "{\n"
        '  "intents": ["intent1", "intent2"]\n'
        "}\n\n"
        f"description: {description}\n"
        f"check_type: {check_type}\n"
    )
    try:
        llm = get_llm(temperature=0.0)
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        data = _extract_json_obj(str(content)) or {}
        intents = data.get("intents", [])
        if isinstance(intents, list):
            valid = {
                "schedule",
                "subtotal",
                "bank_contract_recon",
                "invoice_bank_recon",
                "milestone_bank_recon",
                "generic",
            }
            out = {str(i).strip().lower() for i in intents if str(i).strip().lower() in valid}
            if out:
                return out
    except Exception as e:
        logger.debug("LLM intent classification failed: %s", e)

    # LLM failed; default to generic adjudication only.
    return {"generic"}


async def cross_checker_agent(state: AuditState) -> dict:
    """
    Cross-Checker Agent Node — THE killer feature.
    Iterates through the audit checklist, queries hybrid RAG, detects contradictions.
    """
    writer = get_stream_writer()
    new_steps: list[ReasoningStep] = []
    findings: list[Finding] = []
    documents = state["documents"]
    checklist = state.get("checklist", [])
    doc_ids = [doc.doc_id for doc in documents]
    doc_ids_str = ",".join(doc_ids)

    step = _emit(writer, "thought",
                 f"Starting cross-document analysis. "
                 f"I have {len(checklist)} checks to perform across {len(documents)} document(s). "
                 "I will use both semantic search and graph traversal to find contradictions.")
    new_steps.append(step)

    # ── Phase 1: Graph-level contradiction detection ──────────────────────────
    step = _emit(writer, "tool_call",
                 "Running Neo4j graph analysis to detect connected entities with conflicting values...",
                 tool_name="detect_graph_contradictions",
                 tool_input={"doc_ids": doc_ids_str})
    new_steps.append(step)

    try:
        graph_result = detect_graph_contradictions.invoke({"doc_ids": doc_ids_str})
        graph_data = json.loads(graph_result)
        contradictions = graph_data.get("contradictions", [])

        if contradictions:
            step = _emit(writer, "tool_result",
                         f"Graph analysis found {len(contradictions)} potential contradiction(s)!",
                         tool_name="detect_graph_contradictions",
                         tool_output=f"{len(contradictions)} contradictions")
            new_steps.append(step)

            for c in contradictions[:10]:  # limit to top 10
                raw1 = (c.get("norm1") or c.get("value1") or "").strip()
                raw2 = (c.get("norm2") or c.get("value2") or "").strip()
                doc1 = c.get("doc1_name", c.get("doc1_id", "Doc 1"))
                doc2 = c.get("doc2_name", c.get("doc2_id", "Doc 2"))
                page1 = c.get("page1", 0)
                page2 = c.get("page2", 0)

                if not raw1 or not raw2:
                    logger.debug("Skipping graph pair with empty amount: %s / %s", raw1, raw2)
                    continue
                if parse_monetary_amount(raw1) is None or parse_monetary_amount(raw2) is None:
                    logger.debug("Skipping ambiguous graph amounts: %r / %r", raw1, raw2)
                    continue

                ev1 = f"{doc1}, page {page1}: {c.get('value1', raw1)}"
                ev2 = f"{doc2}, page {page2}: {c.get('value2', raw2)}"
                pair_ctx = _graph_context_for_compare(c, doc1, doc2)

                # LLM first: validate that this is a meaningful like-with-like comparison.
                llm_pair = _llm_assess_amount_pair(
                    raw1, raw2, pair_ctx, [ev1, ev2]
                )
                if not llm_pair:
                    logger.debug("Skipping graph pair due to missing LLM adjudication: %s vs %s", raw1, raw2)
                    continue
                if not llm_pair.get("is_valid_comparison", False):
                    logger.debug("LLM rejected graph pair as invalid comparison: %s vs %s", raw1, raw2)
                    continue

                compare_result = {
                    "is_contradiction": llm_pair.get("is_contradiction", False),
                    "severity": llm_pair.get("severity", "ok"),
                    "confidence": llm_pair.get("confidence", 0.75),
                    "explanation": llm_pair.get("explanation", ""),
                }

                if compare_result.get("is_contradiction"):
                    severity = compare_result.get("severity", "warning")
                    confidence = compare_result.get("confidence", 0.8)
                    explanation = compare_result.get("explanation", "")
                    if severity == "critical" and (
                        not _evidence_line_backs_value(ev1, raw1)
                        or not _evidence_line_backs_value(ev2, raw2)
                    ):
                        severity = "warning"
                        explanation = (
                            "Extraction anomaly / needs review: amounts could not be verified "
                            "against cited snippets; treating as non-critical. " + explanation
                        )

                    step = _emit(writer, "finding",
                                 f"CONTRADICTION DETECTED: {raw1} vs {raw2}\n"
                                 f"Source: {doc1} (p.{page1}) ↔ {doc2} (p.{page2})\n"
                                 f"Severity: {severity.upper()} | Confidence: {confidence:.0%}")
                    new_steps.append(step)

                    findings.append(Finding(
                        severity=severity,
                        title=f"Amount Contradiction: {raw1} vs {raw2}",
                        description=explanation,
                        confidence_score=confidence,
                        source_doc_id=c.get("doc1_id"),
                        source_page=page1,
                        conflicting_doc_id=c.get("doc2_id"),
                        conflicting_page=page2,
                        evidence=[ev1, ev2],
                        recommendation="Verify the correct amount with the contracting parties and request a corrected document.",
                    ))
        else:
            step = _emit(writer, "tool_result",
                         "No direct graph contradictions found. Proceeding with semantic checks.",
                         tool_name="detect_graph_contradictions",
                         tool_output="0 contradictions")
            new_steps.append(step)

    except Exception as e:
        logger.warning("Graph contradiction detection failed: %s", e)
        step = _emit(writer, "thought", f"Graph analysis unavailable ({e}). Continuing with semantic checks.")
        new_steps.append(step)

    # ── Phase 2: Checklist-driven semantic checks ─────────────────────────────
    for i, item in enumerate(checklist):
        step = _emit(writer, "thought",
                     f"[{i+1}/{len(checklist)}] {item.description} [{item.priority.upper()} priority]")
        new_steps.append(step)

        # Build a targeted query for this checklist item
        query = item.description
        item_doc_ids = item.doc_ids_involved or doc_ids
        item_intents = _classify_check_intents(item.description, item.check_type)

        step = _emit(writer, "tool_call",
                     f"Searching across documents: '{query[:80]}...' " if len(query) > 80 else f"Searching: '{query}'",
                     tool_name="search_hybrid_rag",
                     tool_input={"query": query, "doc_ids": ",".join(item_doc_ids)})
        new_steps.append(step)

        try:
            is_bank_recon = bool({
                "bank_contract_recon",
                "invoice_bank_recon",
                "milestone_bank_recon",
            } & item_intents)
            recon_top_k = 20 if is_bank_recon else 8

            search_raw = search_hybrid_rag.invoke({
                "query": query,
                "doc_ids": ",".join(item_doc_ids),
                "top_k": recon_top_k,
            })
            search_data = json.loads(search_raw)
            results = search_data.get("results", [])

            if is_bank_recon:
                bank_doc_ids = [
                    d.doc_id for d in documents if _is_bank_doc(d)
                ]
                if bank_doc_ids:
                    full_bank_chunks = get_all_chunks_for_docs(bank_doc_ids)
                    existing_texts = {r.get("text", "")[:100] for r in results}
                    for chunk in full_bank_chunks:
                        key = (chunk.get("text") or "")[:100]
                        if key and key not in existing_texts:
                            existing_texts.add(key)
                            results.append(chunk)

            step = _emit(writer, "tool_result",
                         f"Found {len(results)} relevant passages "
                         f"({search_data.get('vector_count', 0)} semantic, "
                         f"{search_data.get('graph_count', 0)} graph).",
                         tool_name="search_hybrid_rag",
                         tool_output=f"{len(results)} results")
            new_steps.append(step)

            # LLM-only adjudication over extracted snippets.
            if item.check_type in ("amount_match", "cross_doc_consistency") and len(item_doc_ids) > 1:
                llm_cap = 1 if is_bank_recon else 4
                llm_findings = _llm_adjudicate_check(
                    item, results, documents, max_findings=llm_cap
                )
                if llm_findings:
                    for lf in llm_findings:
                        findings.append(lf)
                        step = _emit(
                            writer,
                            "finding",
                            f"LLM finding: {lf.title} — {lf.severity.upper()}",
                        )
                        new_steps.append(step)
                else:
                    step = _emit(
                        writer,
                        "thought",
                        "LLM review found no reliable contradiction for this check.",
                    )
                    new_steps.append(step)

            elif item.check_type == "date_consistency":
                llm_findings = _llm_adjudicate_check(item, results, documents)
                if llm_findings:
                    for lf in llm_findings:
                        findings.append(lf)
                        step = _emit(
                            writer,
                            "finding",
                            f"LLM date finding: {lf.title} — {lf.severity.upper()}",
                        )
                        new_steps.append(step)
                else:
                    step = _emit(writer, "thought", "LLM date review found no reliable inconsistency.")
                    new_steps.append(step)

        except Exception as e:
            logger.warning("Check failed for item '%s': %s", item.description, e)
            step = _emit(writer, "thought", f"Could not complete check '{item.description[:50]}...': {e}")
            new_steps.append(step)

    # ── Phase 3: External verification for suspicious findings ────────────────
    critical_findings = [f for f in findings if f.severity == "critical"]
    if critical_findings and len(documents) > 0:
        step = _emit(writer, "thought",
                     f"Found {len(critical_findings)} critical finding(s). "
                     "Attempting external verification for key entities...")
        new_steps.append(step)

        # Search for company names if present
        company_entities: list[str] = []
        for doc in documents:
            if doc.parties:
                company_entities.extend(doc.parties[:2])

        for company in company_entities[:2]:
            step = _emit(writer, "tool_call",
                         f"Verifying entity: '{company}'",
                         tool_name="search_web",
                         tool_input={"query": f"company registration {company} Egypt"})
            new_steps.append(step)

            try:
                web_result = search_web.invoke({"query": f"company {company} Egypt commercial registration"})
                web_data = json.loads(web_result)
                step = _emit(writer, "tool_result",
                             web_data.get("summary", "No information found.")[:200],
                             tool_name="search_web",
                             tool_output=web_data.get("summary", "")[:100])
                new_steps.append(step)
            except Exception as e:
                logger.debug("Web search failed: %s", e)

    # ── Phase 4: Deduplicate findings ──────────────────────────────────────────
    deduped: list[Finding] = []
    for f in findings:
        f_topic = _finding_topic_bucket(f)
        f_sig = set(_finding_numeric_signature(f))
        f_docs = set(filter(None, [f.source_doc_id, f.conflicting_doc_id]))
        f_refs = _finding_ref_signature(f)

        merged = False
        for idx, prev in enumerate(deduped):
            p_topic = _finding_topic_bucket(prev)
            p_sig = set(_finding_numeric_signature(prev))
            p_docs = set(filter(None, [prev.source_doc_id, prev.conflicting_doc_id]))
            p_refs = _finding_ref_signature(prev)

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
                merged = True
                logger.debug("Deduped finding %r (duplicate of %r)", f.title, prev.title)
                break

        if not merged:
            deduped.append(f)

    if len(deduped) < len(findings):
        logger.info("Deduplicated findings: %d → %d", len(findings), len(deduped))
    findings = deduped

    # ── Summary ───────────────────────────────────────────────────────────────
    critical = sum(1 for f in findings if f.severity == "critical")
    warnings = sum(1 for f in findings if f.severity == "warning")
    ok_count = sum(1 for f in findings if f.severity == "ok")

    summary = (
        f"Cross-checking complete. "
        f"Results: {critical} critical issue(s), {warnings} warning(s), {ok_count} OK. "
        f"Total findings: {len(findings)}."
    )
    if critical > 0:
        summary += " CRITICAL issues require immediate attention."
    elif warnings > 0:
        summary += " Warnings should be reviewed before finalizing."
    else:
        summary += " Documents appear largely consistent."

    step = _emit(writer, "summary", summary)
    new_steps.append(step)

    return {
        "messages": [AIMessage(content=summary)],
        "findings": findings,
        "reasoning_trace": new_steps,
    }
