"""Cross-Checker Agent orchestration — the core contradiction detection engine."""
import asyncio
import json
import logging

from langgraph.config import get_stream_writer
from langchain_core.messages import AIMessage

from app.models.state import AuditState
from app.models.schemas import ReasoningStep, Finding
from app.tools.hybrid_retriever import search_hybrid_rag
from app.tools.neo4j_tools import detect_graph_contradictions
from app.services.vector_store import get_all_chunks_for_docs
from app.services.redis_store import get_suppressed_signatures
from app.utils.finding_signature import finding_signature
from app.utils.money_parse import parse_monetary_amount
from app.config import get_settings

from app.agents.cross_checker.adjudication import _llm_assess_amount_pair, _llm_adjudicate_check
from app.agents.cross_checker.gates import _accept_finding
from app.agents.cross_checker.dedup import dedupe_findings
from app.agents.cross_checker.evidence import (
    _emit, _evidence_line_backs_value, _evidence_snippets_back_values,
    _graph_context_for_compare, _is_bank_doc, _classify_check_intents,
)

logger = logging.getLogger(__name__)

# Parallel LLM adjudication (Phase 1 pairs, Phase 2 checklist items). Kept small:
# the relation model is the expensive one and OpenRouter rate-limits per key.
_ADJUDICATION_CONCURRENCY = 3


class _StepBuffer:
    """
    Collects reasoning steps from a concurrent worker instead of streaming them.

    Workers run out of order, but the trace the user sees (and the persisted
    reasoning_trace) must stay in checklist order — so each worker buffers its
    steps and the orchestrator flushes them in index order after gathering.
    """

    def __init__(self) -> None:
        self._pending: list[tuple[str, str, dict]] = []

    def emit(self, step_type: str, content: str, **kwargs) -> None:
        self._pending.append((step_type, content, kwargs))

    def flush(self, writer, new_steps: list[ReasoningStep]) -> None:
        for step_type, content, kwargs in self._pending:
            new_steps.append(_emit(writer, step_type, content, **kwargs))


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

    # Every tool below uses a sync driver (Neo4j bolt, Qdrant HTTP) and
    # search_hybrid_rag can make a sync LLM call inside _extract_query_entities.
    # Offloading keeps SSE keepalives and other requests responsive during an audit.
    loop = asyncio.get_running_loop()

    settings = get_settings()
    _gate_kwargs = dict(
        min_evidence_for_critical=settings.finding_min_evidence_for_critical,
        min_confidence_llm=settings.finding_min_confidence_llm,
        min_confidence_graph=settings.finding_min_confidence_graph,
    )
    # Track candidate vs accepted counts for observability
    _candidates_total = 0
    _accepted_total = 0

    # False-positive feedback loop: signatures dismissed by reviewers on prior runs
    # are suppressed here (transparently — we emit a reasoning step when we skip one).
    try:
        suppressed_signatures = await get_suppressed_signatures(state.get("api_key", "default"))
    except Exception as e:
        logger.warning("Could not load suppressed signatures (non-fatal): %s", e)
        suppressed_signatures = set()

    def _is_suppressed(f: Finding) -> bool:
        return bool(suppressed_signatures) and finding_signature(f) in suppressed_signatures

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
        graph_result = await loop.run_in_executor(
            None, detect_graph_contradictions.invoke, {"doc_ids": doc_ids_str}
        )
        graph_data = json.loads(graph_result)
        contradictions = graph_data.get("contradictions", [])

        if contradictions:
            step = _emit(writer, "tool_result",
                         f"Graph analysis found {len(contradictions)} potential contradiction(s)!",
                         tool_name="detect_graph_contradictions",
                         tool_output=f"{len(contradictions)} contradictions")
            new_steps.append(step)

            pair_semaphore = asyncio.Semaphore(_ADJUDICATION_CONCURRENCY)

            async def _adjudicate_pair(c: dict) -> tuple[_StepBuffer, list[Finding], int]:
                """Assess one graph pair. Buffers its steps; returns (buffer, findings, candidates)."""
                buf = _StepBuffer()
                raw1 = (c.get("norm1") or c.get("value1") or "").strip()
                raw2 = (c.get("norm2") or c.get("value2") or "").strip()
                doc1 = c.get("doc1_name", c.get("doc1_id", "Doc 1"))
                doc2 = c.get("doc2_name", c.get("doc2_id", "Doc 2"))
                page1 = c.get("page1", 0)
                page2 = c.get("page2", 0)

                if not raw1 or not raw2:
                    logger.debug("Skipping graph pair with empty amount: %s / %s", raw1, raw2)
                    return buf, [], 0
                if parse_monetary_amount(raw1) is None or parse_monetary_amount(raw2) is None:
                    logger.debug("Skipping ambiguous graph amounts: %r / %r", raw1, raw2)
                    return buf, [], 0

                ev1 = f"{doc1}, page {page1}: {c.get('value1', raw1)}"
                ev2 = f"{doc2}, page {page2}: {c.get('value2', raw2)}"
                pair_ctx = _graph_context_for_compare(c, doc1, doc2)

                # LLM first: validate that this is a meaningful like-with-like comparison.
                async with pair_semaphore:
                    llm_pair = await _llm_assess_amount_pair(raw1, raw2, pair_ctx, [ev1, ev2])
                if not llm_pair:
                    logger.debug("Skipping graph pair due to missing LLM adjudication: %s vs %s", raw1, raw2)
                    return buf, [], 0
                if not llm_pair.get("is_valid_comparison", False):
                    logger.debug("LLM rejected graph pair as invalid comparison: %s vs %s", raw1, raw2)
                    return buf, [], 0

                if not llm_pair.get("is_contradiction", False):
                    return buf, [], 0

                severity = llm_pair.get("severity", "warning")
                confidence = llm_pair.get("confidence", 0.8)
                explanation = llm_pair.get("explanation", "")
                if severity == "critical" and (
                    not _evidence_line_backs_value(ev1, raw1)
                    or not _evidence_line_backs_value(ev2, raw2)
                ):
                    severity = "warning"
                    explanation = (
                        "Extraction anomaly / needs review: amounts could not be verified "
                        "against cited snippets; treating as non-critical. " + explanation
                    )

                candidate = Finding(
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
                )
                accepted, rejection_reason = _accept_finding(
                    candidate, source="graph", **_gate_kwargs
                )
                if accepted and _is_suppressed(candidate):
                    buf.emit("thought",
                             f"Skipping finding suppressed by prior reviewer feedback: "
                             f"{raw1} vs {raw2}")
                    return buf, [], 1
                if accepted:
                    buf.emit("finding",
                             f"CONTRADICTION DETECTED: {raw1} vs {raw2}\n"
                             f"Source: {doc1} (p.{page1}) ↔ {doc2} (p.{page2})\n"
                             f"Severity: {severity.upper()} | Confidence: {confidence:.0%}")
                    return buf, [candidate], 1

                logger.info(
                    "Gate rejected graph finding %r: %s", candidate.title[:60], rejection_reason,
                )
                buf.emit("thought",
                         f"Graph finding filtered out (precision gate: {rejection_reason}): "
                         f"{raw1} vs {raw2}")
                return buf, [], 1

            # Adjudicate the top-10 pairs concurrently, then replay results in the
            # original (materiality-ordered) sequence so the trace stays deterministic.
            pair_results = await asyncio.gather(
                *[_adjudicate_pair(c) for c in contradictions[:10]]
            )
            for buf, pair_findings, candidate_count in pair_results:
                buf.flush(writer, new_steps)
                findings.extend(pair_findings)
                _candidates_total += candidate_count
                _accepted_total += len(pair_findings)
        else:
            step = _emit(writer, "tool_result",
                         "No direct graph contradictions found. Proceeding with semantic checks.",
                         tool_name="detect_graph_contradictions",
                         tool_output="0 contradictions")
            new_steps.append(step)

    except Exception as e:
        # exc_info: this handler previously hid a NameError for the whole phase.
        logger.warning("Graph contradiction detection failed: %s", e, exc_info=True)
        step = _emit(writer, "thought", f"Graph analysis unavailable ({e}). Continuing with semantic checks.")
        new_steps.append(step)

    # ── Phase 2: Checklist-driven semantic checks ─────────────────────────────
    # Pre-fetch all bank-statement chunks once; every bank-reconciliation check below
    # augments its results with the same full set, so re-fetching per item is wasteful.
    bank_doc_ids = [d.doc_id for d in documents if _is_bank_doc(d)]
    full_bank_chunks = (
        await loop.run_in_executor(None, get_all_chunks_for_docs, bank_doc_ids)
        if bank_doc_ids
        else []
    )

    check_semaphore = asyncio.Semaphore(_ADJUDICATION_CONCURRENCY)

    def _collect_llm_findings(
        buf: _StepBuffer,
        llm_findings: list[Finding],
        label: str,
        none_message: str,
    ) -> tuple[list[Finding], int]:
        """
        Gate, suppress, and narrate one adjudication result set.

        Shared by the amount and date branches, which previously carried ~40 lines
        of identical accept/suppress/emit logic each.
        """
        kept: list[Finding] = []
        for lf in llm_findings:
            accepted, rejection_reason = _accept_finding(lf, source="llm", **_gate_kwargs)
            if accepted and _is_suppressed(lf):
                buf.emit("thought",
                         f"Skipping finding suppressed by prior reviewer feedback: {lf.title[:80]}")
                continue
            if accepted:
                kept.append(lf)
                buf.emit("finding", f"{label}: {lf.title} — {lf.severity.upper()}")
            else:
                logger.info(
                    "Gate rejected %s %r: %s", label.lower(), lf.title[:60], rejection_reason,
                )
                buf.emit("thought",
                         f"{label} filtered out (precision gate: {rejection_reason}): "
                         f"{lf.title[:80]}")
        if not llm_findings or not kept:
            buf.emit("thought", none_message)
        return kept, len(llm_findings)

    async def _run_check(i: int, item) -> tuple[_StepBuffer, list[Finding], int]:
        """Retrieve + adjudicate one checklist item. Buffers its steps for ordered replay."""
        buf = _StepBuffer()
        buf.emit("thought",
                 f"[{i+1}/{len(checklist)}] {item.description} [{item.priority.upper()} priority]")

        # Build a targeted query for this checklist item
        query = item.description
        item_doc_ids = item.doc_ids_involved or doc_ids
        item_intents = _classify_check_intents(item.description, item.check_type)

        buf.emit("tool_call",
                 f"Searching across documents: '{query[:80]}...' " if len(query) > 80 else f"Searching: '{query}'",
                 tool_name="search_hybrid_rag",
                 tool_input={"query": query, "doc_ids": ",".join(item_doc_ids)})

        try:
            is_bank_recon = bool({
                "bank_contract_recon",
                "invoice_bank_recon",
                "milestone_bank_recon",
            } & item_intents)
            recon_top_k = 20 if is_bank_recon else 8

            search_raw = await loop.run_in_executor(
                None,
                search_hybrid_rag.invoke,
                {
                    "query": query,
                    "doc_ids": ",".join(item_doc_ids),
                    "top_k": recon_top_k,
                },
            )
            search_data = json.loads(search_raw)
            results = search_data.get("results", [])

            if is_bank_recon and full_bank_chunks:
                existing_texts = {r.get("text", "")[:100] for r in results}
                for chunk in full_bank_chunks:
                    key = (chunk.get("text") or "")[:100]
                    if key and key not in existing_texts:
                        existing_texts.add(key)
                        results.append(chunk)

            buf.emit("tool_result",
                     f"Found {len(results)} relevant passages "
                     f"({search_data.get('vector_count', 0)} semantic, "
                     f"{search_data.get('graph_count', 0)} graph).",
                     tool_name="search_hybrid_rag",
                     tool_output=f"{len(results)} results")

            # LLM-only adjudication over extracted snippets.
            if item.check_type in ("amount_match", "cross_doc_consistency") and len(item_doc_ids) > 1:
                llm_cap = 1 if is_bank_recon else 4
                async with check_semaphore:
                    llm_findings = await _llm_adjudicate_check(
                        item, results, documents, max_findings=llm_cap
                    )
                return (buf, *_collect_llm_findings(
                    buf, llm_findings, "LLM finding",
                    "LLM review found no reliable contradiction for this check.",
                ))

            if item.check_type == "date_consistency":
                async with check_semaphore:
                    llm_findings = await _llm_adjudicate_check(item, results, documents)
                return (buf, *_collect_llm_findings(
                    buf, llm_findings, "LLM date finding",
                    "LLM date review found no reliable inconsistency.",
                ))

            return buf, [], 0

        except Exception as e:
            logger.warning("Check failed for item '%s': %s", item.description, e)
            buf.emit("thought", f"Could not complete check '{item.description[:50]}...': {e}")
            return buf, [], 0

    # Run checklist items concurrently, then replay in checklist order. Deterministic
    # ordering matters: the Phase-3 dedup loop below is order-sensitive.
    check_results = await asyncio.gather(
        *[_run_check(i, item) for i, item in enumerate(checklist)]
    )
    for buf, item_findings, candidate_count in check_results:
        buf.flush(writer, new_steps)
        findings.extend(item_findings)
        _candidates_total += candidate_count
        _accepted_total += len(item_findings)

    # ── Precision gate observability ──────────────────────────────────────────
    logger.info(
        "Precision gate summary: %d candidates → %d accepted (%.0f%% pass rate)",
        _candidates_total,
        _accepted_total,
        (_accepted_total / _candidates_total * 100) if _candidates_total else 0.0,
    )

    # ── Phase 3: Deduplicate findings ──────────────────────────────────────────
    deduped = dedupe_findings(findings)
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
