"""
Conversational Q&A over an audited document set.

A single-shot, retrieval-grounded answerer (not a ReAct loop): it always runs the
hybrid retriever for the question, grounds the model on the retrieved snippets plus
the audit report (findings + reconciliation), and streams the answer token-by-token.
This is more robust than a tool-calling agent for the "why did you flag X / what's
the total paid to Y" questions this UI is for.
"""
import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.models.schemas import AuditReport, AuditSession
from app.tools.hybrid_retriever import search_hybrid_rag
from app.services.redis_store import append_chat_messages
from app.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)

_MAX_SNIPPETS = 12
_MAX_SNIPPET_CHARS = 700
_MAX_FINDINGS_IN_CONTEXT = 25


def _normalize_content(content: Any) -> str:
    """ChatOpenAI chunks may carry str or a list of content parts; flatten to text."""
    if isinstance(content, list):
        return "".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return str(content) if content is not None else ""


def _report_context(report: Optional[AuditReport]) -> str:
    """Compact, model-readable summary of the audit report for grounding."""
    if report is None:
        return "No completed audit report is available yet."
    lines: list[str] = [
        f"Overall risk: {report.overall_risk}.",
        f"Executive summary: {report.executive_summary}".strip(),
    ]
    if report.reconciliation_snapshot:
        rs = report.reconciliation_snapshot
        lines.append(
            "Reconciliation — "
            f"currency {rs.currency}; contract total {rs.contract_total}; "
            f"invoice total {rs.invoice_total}; bank paid {rs.bank_paid_total}; "
            f"variance vs contract {rs.variance_vs_contract}."
        )
    if report.findings:
        lines.append(f"Findings ({len(report.findings)}):")
        for f in report.findings[:_MAX_FINDINGS_IN_CONTEXT]:
            lines.append(
                f"- [{f.severity}] {f.title}: {f.description[:300]} "
                f"(confidence {f.confidence_score:.0%})"
            )
    return "\n".join(lines)


def _retrieve(query: str, doc_ids: list[str]) -> list[dict]:
    """Synchronous hybrid retrieval (offloaded to a thread by the caller)."""
    try:
        raw = search_hybrid_rag.invoke(
            {"query": query, "doc_ids": ",".join(doc_ids), "top_k": _MAX_SNIPPETS}
        )
        data = json.loads(raw)
        return data.get("results", [])
    except Exception as e:
        logger.warning("Q&A retrieval failed: %s", e)
        return []


def _format_snippets(results: list[dict], doc_name_by_id: dict[str, str]) -> tuple[str, list[dict]]:
    """Return (prompt_block, ui_sources) from raw retriever results."""
    prompt_lines: list[str] = []
    ui_sources: list[dict] = []
    for i, r in enumerate(results[:_MAX_SNIPPETS]):
        doc_id = r.get("doc_id", "")
        text = (r.get("text") or r.get("value") or "").strip()
        if not text:
            continue
        name = doc_name_by_id.get(doc_id, doc_id[:8] if doc_id else "document")
        page = int(r.get("page", 0) or 0)
        snippet = text[:_MAX_SNIPPET_CHARS]
        prompt_lines.append(f"[{i + 1}] {name} (page {page}): {snippet}")
        ui_sources.append({"doc_id": doc_id, "doc_name": name, "page": page, "text": snippet})
    return ("\n".join(prompt_lines) or "No relevant passages were found.", ui_sources)


async def answer_audit_question(
    *,
    audit_id: str,
    session: AuditSession,
    report: Optional[AuditReport],
    message: str,
    history: list[dict],
) -> AsyncIterator[dict]:
    """
    Stream a grounded answer to a user question about the audited documents.

    Yields event dicts: {"type": "sources", ...}, {"type": "delta", "text": ...},
    {"type": "done"}, {"type": "error", "message": ...}. Persists the user message
    and assistant answer to the audit's chat history when streaming completes.
    """
    doc_ids = [d.doc_id for d in session.documents]
    doc_name_by_id = {d.doc_id: d.filename for d in session.documents}

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, _retrieve, message, doc_ids)
    snippet_block, ui_sources = _format_snippets(results, doc_name_by_id)
    yield {"type": "sources", "sources": ui_sources}

    system = SystemMessage(content=(
        "You are AuditMind's assistant, answering an auditor's questions about a specific "
        "set of financial documents that have already been analyzed.\n"
        "Ground every answer ONLY in the audit report summary and the document passages "
        "provided below. If the answer is not supported by them, say you don't have enough "
        "information rather than guessing. Cite document names and page numbers when you can. "
        "Be concise and precise with amounts. Reply in the same language as the question "
        "(Arabic or English).\n\n"
        f"=== AUDIT REPORT SUMMARY ===\n{_report_context(report)}\n\n"
        f"=== RETRIEVED DOCUMENT PASSAGES ===\n{snippet_block}\n"
    ))

    messages: list[Any] = [system]
    for turn in history[-8:]:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=message))

    answer_parts: list[str] = []
    try:
        llm = get_llm(temperature=0.2)
        async for chunk in llm.astream(messages):
            text = _normalize_content(getattr(chunk, "content", ""))
            if text:
                answer_parts.append(text)
                yield {"type": "delta", "text": text}
    except Exception as e:
        logger.error("Q&A streaming failed for %s: %s", audit_id, e)
        yield {"type": "error", "message": f"Answer generation failed: {e}"}
        return

    answer = "".join(answer_parts).strip()
    try:
        await append_chat_messages(
            audit_id,
            [{"role": "user", "content": message}, {"role": "assistant", "content": answer}],
        )
    except Exception as e:
        logger.warning("Failed to persist chat history for %s: %s", audit_id, e)

    yield {"type": "done"}
