"""Qdrant vector search tools for LangGraph agents."""
import json
from langchain_core.tools import tool
from app.services.vector_store import semantic_search


@tool
def search_qdrant(query: str, doc_ids: str = "", top_k: int = 8) -> str:
    """
    Semantic search over document chunks stored in Qdrant.

    Args:
        query: Natural language search query (Arabic or English)
        doc_ids: Comma-separated document IDs to restrict search (empty = all docs)
        top_k: Number of results to return (default 8)

    Returns:
        JSON string with matching chunks including text, document ID, and page number
    """
    filter_ids = [d.strip() for d in doc_ids.split(",") if d.strip()] if doc_ids else None

    results = semantic_search(query=query, doc_ids=filter_ids, top_k=top_k)

    if not results:
        return json.dumps({"results": [], "message": "No matching chunks found."})

    return json.dumps({
        "results": [
            {
                "doc_id": r["doc_id"],
                "page": r["page_num"],
                "language": r["language"],
                "score": round(r["score"], 4),
                "text": r["text"][:500] if r["text"] else "",
            }
            for r in results
        ],
        "total": len(results),
    })
