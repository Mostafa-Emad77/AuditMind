"""Master LangGraph StateGraph wiring all AuditMind agents."""
from typing import Literal

from langgraph.graph import StateGraph, START, END
from app.models.state import AuditState
from app.agents.extraction import extraction_agent
from app.agents.planner import planner_agent
from app.agents.cross_checker import cross_checker_agent
from app.agents.report_writer import report_writer_agent


def _route_after_planner(state: AuditState) -> Literal["cross_checker", "report_writer"]:
    """Cross-document contradiction checks need ≥2 documents.

    Single-document audits skip the cross-checker and go straight to the report
    (reconciliation + per-document integrity notes still apply).
    """
    if len(state.get("documents", [])) < 2:
        return "report_writer"
    return "cross_checker"


def build_audit_graph():
    """Build and compile the AuditMind multi-agent graph."""
    builder = StateGraph(AuditState)

    # Register nodes
    builder.add_node("extraction", extraction_agent)
    builder.add_node("planner", planner_agent)
    builder.add_node("cross_checker", cross_checker_agent)
    builder.add_node("report_writer", report_writer_agent)

    # Define the pipeline flow (planner branches: single-doc audits skip cross-checking)
    builder.add_edge(START, "extraction")
    builder.add_edge("extraction", "planner")
    builder.add_conditional_edges("planner", _route_after_planner)
    builder.add_edge("cross_checker", "report_writer")
    builder.add_edge("report_writer", END)

    return builder.compile()


# Singleton compiled graph instance
audit_graph = build_audit_graph()
