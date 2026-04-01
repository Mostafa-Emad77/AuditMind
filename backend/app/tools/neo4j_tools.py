"""Neo4j knowledge graph tools for LangGraph agents."""
import json
from langchain_core.tools import tool
from app.services.graph_builder import (
    find_contradictions,
    query_graph_for_entities,
)


@tool
def search_graph(entity_values: str, doc_ids: str, depth: int = 2) -> str:
    """
    Traverse the Neo4j knowledge graph starting from given entity values.
    Returns related entities found within `depth` hops.

    Args:
        entity_values: Comma-separated entity values to start traversal from
        doc_ids: Comma-separated document IDs to search within
        depth: Graph traversal depth (default 2)

    Returns:
        JSON string with related entities and their document sources
    """
    entities = [e.strip() for e in entity_values.split(",") if e.strip()]
    ids = [d.strip() for d in doc_ids.split(",") if d.strip()]

    if not entities or not ids:
        return json.dumps({"error": "entity_values and doc_ids are required"})

    results = query_graph_for_entities(entities, ids, depth=depth)

    return json.dumps({
        "related_entities": results,
        "total": len(results),
    })


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
