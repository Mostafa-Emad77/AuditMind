"""Master LangGraph StateGraph wiring all AuditMind agents."""
from typing import Literal

from langgraph.graph import StateGraph, START, END
from app.models.state import AuditState
from app.agents.extraction import extraction_agent
from app.agents.planner import planner_agent
from app.agents.cross_checker import cross_checker_agent
from app.agents.report_writer import report_writer_agent


# Normal LLM checklists run 6-12 items; anything under this likely came from the
# fallback path (planner_tools) or a degenerate generation — worth one deeper pass.
_REPLAN_CHECKLIST_THIN = 4
# Planner passes total: 1 initial + 1 re-plan. Bounds the loop; termination guaranteed.
_REPLAN_MAX_PASSES = 2


def _route_after_planner(state: AuditState) -> Literal["cross_checker", "report_writer"]:
    """Cross-document contradiction checks need ≥2 documents.

    Single-document audits skip the cross-checker and go straight to the report
    (reconciliation + per-document integrity notes still apply).
    """
    if len(state.get("documents", [])) < 2:
        return "report_writer"
    return "cross_checker"


def _route_after_cross_checker(state: AuditState) -> Literal["report_writer", "planner"]:
    """Empty findings on a thin plan get one deeper re-plan pass.

    Guards: needs ≥2 docs (single-doc audits never enter the cross-checker),
    a thin checklist, zero findings, and a remaining retry.
    """
    if (
        len(state.get("documents", [])) >= 2
        and len(state.get("findings", [])) == 0
        and len(state.get("checklist", [])) < _REPLAN_CHECKLIST_THIN
        and state.get("replan_count", 0) < _REPLAN_MAX_PASSES
    ):
        return "planner"
    return "report_writer"


def build_audit_graph():
    """Build and compile the AuditMind multi-agent graph."""
    builder = StateGraph(AuditState)

    # Register nodes
    builder.add_node("extraction", extraction_agent)
    builder.add_node("planner", planner_agent)
    builder.add_node("cross_checker", cross_checker_agent)
    builder.add_node("report_writer", report_writer_agent)

    # Pipeline flow: planner branches on doc count; cross-checker loops back
    # to the planner at most once (empty findings + thin plan).
    builder.add_edge(START, "extraction")
    builder.add_edge("extraction", "planner")
    builder.add_conditional_edges("planner", _route_after_planner)
    builder.add_conditional_edges("cross_checker", _route_after_cross_checker)
    builder.add_edge("report_writer", END)

    return builder.compile()


# Singleton compiled graph instance
audit_graph = build_audit_graph()
