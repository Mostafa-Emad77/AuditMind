"""Master LangGraph StateGraph wiring all AuditMind agents."""
from langgraph.graph import StateGraph, START, END
from app.models.state import AuditState
from app.agents.extraction import extraction_agent
from app.agents.planner import planner_agent
from app.agents.cross_checker import cross_checker_agent
from app.agents.report_writer import report_writer_agent


def build_audit_graph():
    """Build and compile the AuditMind multi-agent graph."""
    builder = StateGraph(AuditState)

    # Register nodes
    builder.add_node("extraction", extraction_agent)
    builder.add_node("planner", planner_agent)
    builder.add_node("cross_checker", cross_checker_agent)
    builder.add_node("report_writer", report_writer_agent)

    # Define the pipeline flow
    builder.add_edge(START, "extraction")
    builder.add_edge("extraction", "planner")
    builder.add_edge("planner", "cross_checker")
    builder.add_edge("cross_checker", "report_writer")
    builder.add_edge("report_writer", END)

    return builder.compile()


# Singleton compiled graph instance
audit_graph = build_audit_graph()
