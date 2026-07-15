"""Neo4j knowledge graph tools for LangGraph agents."""
import json
from langchain_core.tools import tool
from app.services.graph_builder import find_contradictions


@tool
def detect_graph_contradictions(doc_ids: str) -> str:
    """
    Find amount contradictions between documents in the knowledge graph.
    Looks for entities that are connected through the graph but have conflicting values.

    Args:
        doc_ids: Comma-separated document IDs to check for contradictions

    Returns:
        JSON string listing all detected contradictions with source citations
    """
    ids = [d.strip() for d in doc_ids.split(",") if d.strip()]

    if not ids:
        return json.dumps({"error": "doc_ids is required"})

    contradictions = find_contradictions(ids)

    return json.dumps({
        "contradictions": contradictions,
        "total": len(contradictions),
        "has_contradictions": len(contradictions) > 0,
    })
