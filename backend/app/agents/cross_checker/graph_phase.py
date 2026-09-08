"""Phase 1: Neo4j graph-pair contradiction detection + LLM adjudication."""
import asyncio
import json
import logging

from app.models.schemas import Finding
from app.utils.money_parse import parse_monetary_amount
from app.agents.cross_checker.context import (
    _ADJUDICATION_CONCURRENCY,
    _StepBuffer,
    CrossCheckContext,
)
from app.agents.cross_checker.evidence import (
    _emit,
    _evidence_line_backs_value,
    _graph_context_for_compare,
)

logger = logging.getLogger(__name__)


def _dedupe_graph_pairs(contradictions: list[dict]) -> list[dict]:
    """One row per (numeric pair, reason): duplicates across doc combinations would
    otherwise crowd real pairs out of the top-N slice."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for c in contradictions:
        v1, v2 = c.get("amount_value1"), c.get("amount_value2")
        num_key = frozenset({
            round(float(v1), 2) if v1 is not None else str(c.get("norm1") or "").lower(),
            round(float(v2), 2) if v2 is not None else str(c.get("norm2") or "").lower(),
        })
        key = (num_key, str(c.get("comparison_reason") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    if len(out) < len(contradictions):
        logger.info("Graph pairs deduplicated: %d → %d", len(contradictions), len(out))
    return out


async def run_graph_phase(
    ctx: CrossCheckContext,
    *,
    detect_fn,
    assess_fn,
    accept_fn,
) -> list[Finding]:
    """Phase 1: graph-level contradiction detection. Returns accepted findings."""
    findings: list[Finding] = []
    writer = ctx.writer
    doc_ids_str = ",".join(ctx.doc_ids)

    step = _emit(writer, "tool_call",
                 "Running Neo4j graph analysis to detect connected entities with conflicting values...",
                 tool_name="detect_graph_contradictions",
                 tool_input={"doc_ids": doc_ids_str})
    ctx.new_steps.append(step)

    try:
        graph_result = await ctx.loop.run_in_executor(
            None, detect_fn.invoke, {"doc_ids": doc_ids_str}
        )
        graph_data = json.loads(graph_result)
        contradictions = _dedupe_graph_pairs(graph_data.get("contradictions", []))

        if contradictions:
            step = _emit(writer, "tool_result",
                         f"Graph analysis found {len(contradictions)} potential contradiction(s)!",
                         tool_name="detect_graph_contradictions",
                         tool_output=f"{len(contradictions)} contradictions")
            ctx.new_steps.append(step)

            pair_semaphore = asyncio.Semaphore(_ADJUDICATION_CONCURRENCY)

            async def _adjudicate_pair(c: dict) -> tuple[_StepBuffer, list[Finding], int]:
                """Assess one graph pair. Buffers its steps; returns (buffer, findings, candidates)."""
                buf = _StepBuffer()
                raw1 = (c.get("norm1") or c.get("value1") or "").strip()
                raw2 = (c.get("norm2") or c.get("value2") or "").strip()
                # Never expose the raw doc_id UUID to the LLM or evidence text.
                doc1 = c.get("doc1_name") or "Document 1"
                doc2 = c.get("doc2_name") or "Document 2"
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
                    llm_pair = await assess_fn(raw1, raw2, pair_ctx, [ev1, ev2])
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
                accepted, rejection_reason = accept_fn(
                    candidate, source="graph", **ctx.gate_kwargs
                )
                if accepted and ctx.is_suppressed(candidate):
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

            # Adjudicate top-10 concurrently; replay in materiality order.
            pair_results = await asyncio.gather(
                *[_adjudicate_pair(c) for c in contradictions[:10]]
            )
            for buf, pair_findings, candidate_count in pair_results:
                buf.flush(writer, ctx.new_steps)
                findings.extend(pair_findings)
                ctx.candidates_total += candidate_count
                ctx.accepted_total += len(pair_findings)
        else:
            step = _emit(writer, "tool_result",
                         "No direct graph contradictions found. Proceeding with semantic checks.",
                         tool_name="detect_graph_contradictions",
                         tool_output="0 contradictions")
            ctx.new_steps.append(step)

    except Exception as e:
        logger.warning("Graph contradiction detection failed: %s", e, exc_info=True)
        step = _emit(writer, "thought", f"Graph analysis unavailable ({e}). Continuing with semantic checks.")
        ctx.new_steps.append(step)

    return findings
