"""Phase 2: checklist-driven semantic checks via hybrid RAG + LLM adjudication."""
import asyncio
import json
import logging

from app.models.schemas import Finding
from app.agents.cross_checker.context import (
    _ADJUDICATION_CONCURRENCY,
    _StepBuffer,
    CrossCheckContext,
)
from app.agents.cross_checker.evidence import _classify_check_intents, _is_bank_doc

logger = logging.getLogger(__name__)


def _collect_llm_findings(
    ctx: CrossCheckContext,
    buf: _StepBuffer,
    llm_findings: list[Finding],
    label: str,
    none_message: str,
    *,
    accept_fn,
) -> tuple[list[Finding], int]:
    """Gate, suppress, and narrate one adjudication result set."""
    kept: list[Finding] = []
    for lf in llm_findings:
        accepted, rejection_reason = accept_fn(lf, source="llm", **ctx.gate_kwargs)
        if accepted and ctx.is_suppressed(lf):
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


async def run_checklist_phase(
    ctx: CrossCheckContext,
    *,
    search_fn,
    get_chunks_fn,
    adjudicate_fn,
    accept_fn,
) -> list[Finding]:
    """Run concurrently, replay in checklist order (dedup is order-sensitive)."""
    findings: list[Finding] = []
    checklist = ctx.checklist
    doc_ids = ctx.doc_ids

    # Bank chunks fetched once; every bank-recon check appends the full set.
    bank_doc_ids = [d.doc_id for d in ctx.documents if _is_bank_doc(d)]
    full_bank_chunks = (
        await ctx.loop.run_in_executor(None, get_chunks_fn, bank_doc_ids)
        if bank_doc_ids
        else []
    )

    check_semaphore = asyncio.Semaphore(_ADJUDICATION_CONCURRENCY)

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

            search_raw = await ctx.loop.run_in_executor(
                None,
                search_fn.invoke,
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
                    llm_findings = await adjudicate_fn(
                        item, results, ctx.documents, max_findings=llm_cap
                    )
                return (buf, *_collect_llm_findings(
                    ctx, buf, llm_findings, "LLM finding",
                    "LLM review found no reliable contradiction for this check.",
                    accept_fn=accept_fn,
                ))

            if item.check_type == "self_reference":
                # Handled deterministically in Phase 1c; recorded for the trace.
                buf.emit("thought",
                         "Self-reference validation is evaluated deterministically from the "
                         "knowledge graph (see 'Reference validation' above); no LLM pass needed.")
                return buf, [], 0

            if item.check_type == "date_consistency":
                async with check_semaphore:
                    llm_findings = await adjudicate_fn(item, results, ctx.documents)
                return (buf, *_collect_llm_findings(
                    ctx, buf, llm_findings, "LLM date finding",
                    "LLM date review found no reliable inconsistency.",
                    accept_fn=accept_fn,
                ))

            return buf, [], 0

        except Exception as e:
            logger.warning("Check failed for item '%s': %s", item.description, e)
            buf.emit("thought", f"Could not complete check '{item.description[:50]}...': {e}")
            return buf, [], 0

    check_results = await asyncio.gather(
        *[_run_check(i, item) for i, item in enumerate(checklist)]
    )
    for buf, item_findings, candidate_count in check_results:
        buf.flush(ctx.writer, ctx.new_steps)
        findings.extend(item_findings)
        ctx.candidates_total += candidate_count
        ctx.accepted_total += len(item_findings)

    return findings
