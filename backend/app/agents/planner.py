"""Audit Planner Agent — generates a custom audit checklist."""
import json
import logging
from langgraph.config import get_stream_writer
from langchain_core.messages import AIMessage

from app.models.state import AuditState
from app.models.schemas import ReasoningStep, ChecklistItem
from app.tools.planner_tools import generate_checklist

logger = logging.getLogger(__name__)


def _emit(writer, step_type: str, content: str, **kwargs) -> ReasoningStep:
    step = ReasoningStep(agent="planner", step_type=step_type, content=content, **kwargs)
    writer({"type": "reasoning_step", "step": step.model_dump(mode="json")})
    return step


async def planner_agent(state: AuditState) -> dict:
    """
    Audit Planner Agent Node.
    Analyzes document types and generates a targeted audit checklist.
    """
    writer = get_stream_writer()
    new_steps: list[ReasoningStep] = []
    documents = state["documents"]

    step = _emit(writer, "thought",
                 "Starting audit planning. I'll analyze what document types are present "
                 "and create a targeted checklist of things to verify.")
    new_steps.append(step)

    if not documents:
        step = _emit(writer, "summary", "No documents to plan for.")
        new_steps.append(step)
        return {"checklist": [], "reasoning_trace": new_steps}

    # Analyze document types
    doc_types = [doc.doc_type for doc in documents]
    doc_ids = [doc.doc_id for doc in documents]
    type_counts = {}
    for t in doc_types:
        type_counts[t] = type_counts.get(t, 0) + 1

    type_summary = ", ".join(f"{count}x {t.replace('_', ' ')}" for t, count in type_counts.items())
    step = _emit(writer, "thought",
                 f"Document inventory: {type_summary}. "
                 f"Languages: {', '.join(set(doc.language for doc in documents))}.")
    new_steps.append(step)

    if len(documents) < 2:
        step = _emit(writer, "thought",
                      "Single-document audit — there is nothing to cross-check against, "
                      "so the cross-checker phase will be skipped and I'll go straight "
                      "to the report.")
        new_steps.append(step)

    # Re-plan pass: the checklist from the prior pass is already in state.
    prior_checklist: list[ChecklistItem] = list(state.get("checklist", []) or [])
    is_replan = len(prior_checklist) > 0
    if is_replan:
        step = _emit(writer, "thought",
                      f"Re-plan pass: the first sweep found no contradictions with "
                      f"{len(prior_checklist)} checks. I'll generate deeper follow-up "
                      f"checks and drop anything already covered.")
        new_steps.append(step)

    # Determine what pairs/combinations exist and explain reasoning
    reasoning_parts = []
    if "invoice" in doc_types and "contract" in doc_types:
        reasoning_parts.append(
            "Invoice + Contract pair detected → I will check: "
            "amounts match, party names consistent, dates within contract period, VAT correct."
        )
    if "bank_statement" in doc_types:
        reasoning_parts.append(
            "Bank statement present → I will cross-reference payment amounts and dates."
        )
    if "balance_sheet" in doc_types:
        reasoning_parts.append(
            "Balance sheet present → I will verify assets = liabilities + equity, check unusual entries."
        )
    if "audit_report" in doc_types:
        reasoning_parts.append(
            "Audit report present → I will verify cited figures match source documents."
        )
    if not reasoning_parts:
        reasoning_parts.append(
            "General document audit → I will check for cross-document amount consistency, "
            "date logic, and duplicate identifiers."
        )

    for part in reasoning_parts:
        step = _emit(writer, "thought", part)
        new_steps.append(step)

    # Generate checklist using tool
    step = _emit(writer, "tool_call",
                 "Generating audit checklist based on document types...",
                 tool_name="generate_checklist",
                 tool_input={"doc_types": doc_types, "doc_ids": doc_ids})
    new_steps.append(step)

    try:
        result = generate_checklist.invoke({
            "doc_types_json": json.dumps(doc_types),
            "doc_ids_json": json.dumps(doc_ids),
        })
        result_data = json.loads(result)
        checklist_dicts = result_data.get("checklist", [])

        # Parse into ChecklistItem objects
        checklist: list[ChecklistItem] = []
        for item_dict in checklist_dicts:
            try:
                checklist.append(ChecklistItem(**item_dict))
            except Exception:
                continue

        step = _emit(writer, "tool_result",
                     f"Generated {len(checklist)} audit checks across {len(doc_types)} documents.",
                     tool_name="generate_checklist",
                     tool_output=f"{len(checklist)} checklist items")
        new_steps.append(step)

        # Show checklist summary
        high_priority = [c for c in checklist if c.priority == "high"]
        checklist_preview = "\n".join(
            f"  [{c.priority.upper()}] {c.description}" for c in checklist[:8]
        )
        if is_replan:
            seen = {c.description.strip().lower() for c in prior_checklist}
            fresh = [c for c in checklist if c.description.strip().lower() not in seen]
            step = _emit(writer, "thought",
                         f"Re-plan: {len(fresh)} new checks added"
                         + (f" ({len(checklist) - len(fresh)} duplicates of prior checks dropped)."
                            if len(fresh) < len(checklist) else "."))
            new_steps.append(step)
            checklist = [*prior_checklist, *fresh]

        step = _emit(writer, "summary",
                     f"Audit plan ready. {len(high_priority)} high-priority checks, "
                     f"{len(checklist) - len(high_priority)} standard checks.\n\n"
                     f"Checks to perform:\n{checklist_preview}"
                     + (f"\n  ... and {len(checklist) - 8} more" if len(checklist) > 8 else ""))
        new_steps.append(step)

        return {
            "messages": [AIMessage(content=f"Audit checklist generated: {len(checklist)} items.")],
            "checklist": checklist,
            "replan_count": 1,
            "reasoning_trace": new_steps,
        }

    except Exception as e:
        logger.error("Planner failed to generate checklist: %s", e)
        step = _emit(writer, "summary", f"Checklist generation failed: {e}. Proceeding with universal checks.")
        new_steps.append(step)

        # Fallback minimal checklist
        fallback = [
            ChecklistItem(
                description="Check for cross-document amount contradictions",
                check_type="cross_doc_consistency",
                doc_ids_involved=doc_ids,
                priority="high",
            ),
            ChecklistItem(
                description="Verify date consistency across all documents",
                check_type="date_consistency",
                doc_ids_involved=doc_ids,
                priority="medium",
            ),
        ]
        if is_replan:
            seen = {c.description.strip().lower() for c in prior_checklist}
            fallback = [c for c in fallback if c.description.strip().lower() not in seen]
            checklist = [*prior_checklist, *fallback]
        else:
            checklist = fallback
        return {
            "messages": [AIMessage(content="Using fallback checklist due to error.")],
            "checklist": checklist,
            "replan_count": 1,
            "reasoning_trace": new_steps,
        }
