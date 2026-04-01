"""
Hybrid retriever: combines Qdrant semantic search + Neo4j graph traversal.
This is the core retrieval mechanism for the Cross-Checker agent.
"""
import json
import re
from langchain_core.tools import tool
from app.services.vector_store import semantic_search
from app.services.graph_builder import query_graph_for_entities
from app.utils.arabic_normalizer import extract_amounts
from app.utils.llm_factory import get_llm

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_query_entities(query: str) -> list[str]:
    """Extract graph-traversal entities using LLM, with safe regex fallback."""
    q = (query or "").strip()
    if not q:
        return []

    prompt = (
        "Extract traversal entities from this audit query.\n"
        "Include company names, contract IDs, invoice IDs, monetary amounts, and anchor terms.\n"
        "Return JSON only:\n"
        "{\n"
        '  "entities": ["entity1", "entity2"]\n'
        "}\n"
        f"query: {q}\n"
    )
    try:
        llm = get_llm(temperature=0.0)
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        data = _extract_json_dict(str(content)) or {}
        ents = data.get("entities", [])
        if isinstance(ents, list):
            cleaned = []
            seen = set()
            for e in ents:
                s = str(e).strip()
                if not s:
                    continue
                k = s.lower()
                if k in seen:
                    continue
                seen.add(k)
                cleaned.append(s)
                if len(cleaned) >= 12:
                    break
            if cleaned:
                return cleaned
    except Exception:
        pass

    # Fallback: minimal regex extraction to avoid empty graph route.
    entities = []
    amount_matches = extract_amounts(q)
    entities.extend(a["raw"] for a in amount_matches)
    words = re.findall(r"\b[A-Z][a-zA-Z]+\b|\b\d{4,}\b", q)
    entities.extend(words[:5])
    arabic_words = re.findall(r"[\u0600-\u06FF]{3,}", q)
    entities.extend(arabic_words[:5])
    return list(dict.fromkeys(entities))


def _extract_json_dict(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    m = _JSON_OBJ_RE.search(text or "")
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _route_query(query: str) -> tuple[str, str, str]:
    """
    Decide retrieval path:
      - vector: factual / semantic lookup
      - graph: relational / cross-document checks
    Returns: (route, reason, router_source)
    """
    q = (query or "").strip()
    if not q:
        return "vector", "Empty query defaults to semantic search.", "fallback"

    # LLM-only router.
    prompt = (
        "Route this audit retrieval query to ONE backend.\n"
        "Backends:\n"
        "- vector: factual/semantic text lookup within documents\n"
        "- graph: relational/cross-document traversal (entity links, contradictions, reconciliation)\n\n"
        f"Query: {q}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "route": "vector|graph",\n'
        '  "reason": "short reason"\n'
        "}\n"
    )
    try:
        llm = get_llm(temperature=0.0)
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        data = _extract_json_dict(str(content))
        route = str((data or {}).get("route", "vector")).strip().lower()
        reason = str((data or {}).get("reason", "")).strip() or "LLM-routed query."
        if route not in ("vector", "graph"):
            route = "vector"
        return route, reason, "llm"
    except Exception:
        return "vector", "Router fallback to semantic search.", "fallback"


@tool
def search_hybrid_rag(query: str, doc_ids: str, top_k: int = 10) -> str:
    """
    Hybrid retrieval combining Qdrant semantic search AND Neo4j graph traversal.
    This is the primary tool for finding relevant information across all documents.

    Use this when checking consistency, finding specific clauses, or looking up values.

    Args:
        query: Natural language query describing what information you need
        doc_ids: Comma-separated document IDs to search across
        top_k: Number of semantic results to return (graph results are additional)

    Returns:
        JSON string with merged results from both vector and graph search
    """
    ids = [d.strip() for d in doc_ids.split(",") if d.strip()]

    # 1) Router picks retrieval backend for this query.
    route, route_reason, routed_by = _route_query(query)

    # 2) Execute selected branch.
    vector_results = []
    graph_results = []
    effective_route = route

    if route == "vector":
        vector_results = semantic_search(query=query, doc_ids=ids if ids else None, top_k=top_k)
    else:
        if ids:
            query_entities = _extract_query_entities(query)
            if query_entities:
                graph_results = query_graph_for_entities(query_entities, ids, depth=2)
            else:
                # Graph route with no entities to traverse → safe fallback.
                vector_results = semantic_search(query=query, doc_ids=ids if ids else None, top_k=top_k)
                effective_route = "vector"
                route_reason = "Graph route had no extractable entities; used semantic fallback."
        else:
            vector_results = semantic_search(query=query, doc_ids=None, top_k=top_k)
            effective_route = "vector"
            route_reason = "Graph traversal requires doc_ids; used semantic fallback."

    # 3) Merge and deduplicate
    seen_texts = set()
    merged = []

    for r in vector_results:
        key = r.get("text", "")[:100]
        if key not in seen_texts:
            seen_texts.add(key)
            merged.append({
                "source": "vector",
                "doc_id": r["doc_id"],
                "page": r["page_num"],
                "language": r["language"],
                "score": round(r["score"], 4),
                "text": r["text"] if r["text"] else "",
            })

    for r in graph_results:
        key = r.get("value", "")[:100]
        if key not in seen_texts:
            seen_texts.add(key)
            merged.append({
                "source": "graph",
                "doc_id": r.get("doc_id", ""),
                "page": r.get("page", 0),
                "entity_type": r.get("entity_type", ""),
                "value": r.get("value", ""),
                "normalized_value": r.get("normalized_value", ""),
                "via_relationship": r.get("via_relationship", ""),
            })

    return json.dumps({
        "route": effective_route,
        "route_reason": route_reason,
        "router": routed_by,
        "results": merged,
        "vector_count": len(vector_results),
        "graph_count": len(graph_results),
        "total": len(merged),
    })
