"""Neo4j knowledge graph tools for LangGraph agents."""
import json
from langchain_core.tools import tool
from app.services.graph_builder import find_contradictions


@tool
def detect_graph_contradictions(doc_ids: str) -> str:
    """Amount contradictions between graph-linked entities (doc_ids comma-separated) → JSON."""
    ids = [d.strip() for d in doc_ids.split(",") if d.strip()]

    if not ids:
        return json.dumps({"error": "doc_ids is required"})

    contradictions = find_contradictions(ids)

    return json.dumps({
        "contradictions": contradictions,
        "total": len(contradictions),
        "has_contradictions": len(contradictions) > 0,
    })
