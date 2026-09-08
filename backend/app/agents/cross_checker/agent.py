"""Cross-Checker Agent orchestration — thin LangGraph node wiring the phases.

Phases (each in its own module, independently testable):
  Phase 1   graph_phase.run_graph_phase — Neo4j pairs + LLM adjudication
  Phase 1c  reference_phase.run_reference_phase — deterministic restatement check
  Phase 1b  signatory_phase.run_signatory_phase — amendment authority check
  Phase 2   checklist_phase.run_checklist_phase — hybrid RAG + LLM per checklist item
  Phase 3   finalize.finalize_findings — dedup + hallucinated-UUID guard

This module keeps the historical names (service handles, gates, adjudicators,
_StepBuffer, _dedupe_graph_pairs, _ADJUDICATION_CONCURRENCY) at module level so
existing tests that monkeypatch ``app.agents.cross_checker.agent`` keep working:
the orchestrator resolves them lazily and passes them into the phases.
"""
import asyncio
import logging

from langgraph.config import get_stream_writer
from langchain_core.messages import AIMessage

from app.models.state import AuditState
from app.models.schemas import ReasoningStep, Finding
from app.tools.hybrid_retriever import search_hybrid_rag
from app.tools.neo4j_tools import detect_graph_contradictions
from app.services.graph_builder import find_reference_mismatches, find_signatory_mismatches
from app.services.vector_store import get_all_chunks_for_docs
from app.services.redis_store import get_suppressed_signatures
from app.utils.finding_signature import finding_signature
from app.utils.money_parse import parse_monetary_amount  # noqa: F401 — re-exported for compat
from app.config import get_settings

from app.agents.cross_checker.adjudication import _llm_assess_amount_pair, _llm_adjudicate_check
from app.agents.cross_checker.gates import _accept_finding
from app.agents.cross_checker.dedup import dedupe_findings  # noqa: F401 — re-exported for compat
from app.agents.cross_checker.validation import drop_hallucinated_identifier_findings  # noqa: F401
from app.agents.cross_checker.evidence import (
    _emit, _evidence_line_backs_value, _graph_context_for_compare, _is_bank_doc, _classify_check_intents,
)  # noqa: F401 — re-exported for compat

# Phase internals (public for direct unit testing of individual phases).
from app.agents.cross_checker.context import (
    _ADJUDICATION_CONCURRENCY,
    _StepBuffer,
    CrossCheckContext,
)
from app.agents.cross_checker.graph_phase import _dedupe_graph_pairs, run_graph_phase
from app.agents.cross_checker.reference_phase import run_reference_phase
from app.agents.cross_checker.signatory_phase import run_signatory_phase
from app.agents.cross_checker.checklist_phase import run_checklist_phase
from app.agents.cross_checker.finalize import finalize_findings, build_summary

logger = logging.getLogger(__name__)


async def cross_checker_agent(state: AuditState) -> dict:
    """
    Cross-Checker Agent Node — THE killer feature.
    Iterates through the audit checklist, queries hybrid RAG, detects contradictions.
    """
    writer = get_stream_writer()
    documents = state["documents"]
    checklist = state.get("checklist", [])
    doc_ids = [doc.doc_id for doc in documents]

    # Tools use sync drivers; offload so SSE keepalives stay responsive.
    loop = asyncio.get_running_loop()

    settings = get_settings()
    gate_kwargs = dict(
        min_evidence_for_critical=settings.finding_min_evidence_for_critical,
        min_confidence_llm=settings.finding_min_confidence_llm,
        min_confidence_graph=settings.finding_min_confidence_graph,
    )

    # Signatures dismissed by reviewers on prior runs are suppressed (with a trace step).
    try:
        suppressed_signatures = await get_suppressed_signatures(state.get("api_key", "default"))
    except Exception as e:
        logger.warning("Could not load suppressed signatures (non-fatal): %s", e)
        suppressed_signatures = set()

    def _is_suppressed(f: Finding) -> bool:
        return bool(suppressed_signatures) and finding_signature(f) in suppressed_signatures

    ctx = CrossCheckContext(
        writer=writer,
        loop=loop,
        documents=documents,
        doc_ids=doc_ids,
        checklist=checklist,
        gate_kwargs=gate_kwargs,
        is_suppressed=_is_suppressed,
    )

    step = _emit(writer, "thought",
                 f"Starting cross-document analysis. "
                 f"I have {len(checklist)} checks to perform across {len(documents)} document(s). "
                 "I will use both semantic search and graph traversal to find contradictions.")
    ctx.new_steps.append(step)

    findings: list[Finding] = []
    # Resolve via module globals at call time so monkeypatched test doubles flow through.
    findings.extend(await run_graph_phase(
        ctx,
        detect_fn=detect_graph_contradictions,
        assess_fn=_llm_assess_amount_pair,
        accept_fn=_accept_finding,
    ))
    findings.extend(await run_reference_phase(
        ctx, find_fn=find_reference_mismatches, accept_fn=_accept_finding,
    ))
    findings.extend(await run_signatory_phase(
        ctx, find_fn=find_signatory_mismatches, accept_fn=_accept_finding,
    ))
    findings.extend(await run_checklist_phase(
        ctx,
        search_fn=search_hybrid_rag,
        get_chunks_fn=get_all_chunks_for_docs,
        adjudicate_fn=_llm_adjudicate_check,
        accept_fn=_accept_finding,
    ))

    findings = finalize_findings(ctx, findings)
    summary = build_summary(ctx, findings)

    return {
        "messages": [AIMessage(content=summary)],
        "findings": findings,
        "reasoning_trace": ctx.new_steps,
    }


__all__ = [
    "cross_checker_agent",
    "CrossCheckContext",
    "run_graph_phase",
    "run_reference_phase",
    "run_signatory_phase",
    "run_checklist_phase",
    "finalize_findings",
    "build_summary",
    "_dedupe_graph_pairs",
    "_StepBuffer",
    "_ADJUDICATION_CONCURRENCY",
]
