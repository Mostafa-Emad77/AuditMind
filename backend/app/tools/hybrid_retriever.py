"""
Hybrid retriever: combines Qdrant semantic search + Neo4j graph traversal.
This is the core retrieval mechanism for the Cross-Checker agent.
"""
import json
import re
from langchain_core.tools import tool
from app.services.vector_store import semantic_search, keyword_search
from app.services.graph_builder import query_graph_for_entities
from app.utils.arabic_normalizer import extract_amounts
from app.utils.llm_factory import get_llm
from app.utils.llm_json import extract_json_obj as _extract_json_dict, normalize_llm_content

# Identifier-like tokens (contract IDs, invoice numbers) that dense embeddings miss.
# Examples matched: INV-2024-0837, C/MOH/2023/041, CON.21.A, 2024-INV-9
_ID_TOKEN_RE = re.compile(r"\b[A-Z0-9][A-Z0-9\-_/.]{3,}\b")


def _extract_id_like_tokens(query: str) -> list[str]:
    """Extract identifier-shaped tokens from a query for keyword_search."""
    if not query:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for m in _ID_TOKEN_RE.finditer(query):
        tok = m.group(0)
        # Require at least one digit OR a structural separator to avoid plain words.
        if not (any(ch.isdigit() for ch in tok) or any(ch in "-_/." for ch in tok)):
            continue
        k = tok.lower()
        if k in seen:
            continue
        seen.add(k)
        tokens.append(tok)
        if len(tokens) >= 5:
            break
    return tokens


def _regex_query_entities(q: str) -> list[str]:
    """Deterministic entity pull: amounts, capitalised terms, long numbers, Arabic words."""
    entities = []
    amount_matches = extract_amounts(q)
    entities.extend(a["raw"] for a in amount_matches)
    words = re.findall(r"\b[A-Z][a-zA-Z]+\b|\b\d{4,}\b", q)
    entities.extend(words[:5])
    arabic_words = re.findall(r"[؀-ۿ]{3,}", q)
    entities.extend(arabic_words[:5])
    return list(dict.fromkeys(entities))


def _extract_query_entities(query: str) -> list[str]:
    """Graph-traversal entities for a query. Regex by default; LLM path is opt-in
    (it cost one slow call per checklist item for no extra recall)."""
    from app.config import get_settings

    q = (query or "").strip()
    if not q:
        return []
    if not get_settings().retriever_llm_query_entities:
        return _regex_query_entities(q)

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
        content = normalize_llm_content(resp)
        data = _extract_json_dict(content) or {}
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
    return _regex_query_entities(q)


# Relational signals that add graph traversal (rule-based, no LLM).
_GRAPH_ROUTE_KEYWORDS = (
    "reconcil", "contradict", "cross-document", "cross document",
    "across documents", "between documents", "versus", " vs ",
    "discrepanc", "mismatch", "linked", "traceab",
)


def _route_query(query: str) -> tuple[str, str, str]:
    """Rule-based route: graph for relational/cross-doc checks, else vector."""
    q = (query or "").strip()
    if not q:
        return "vector", "Empty query defaults to semantic search.", "fallback"

    ql = q.lower()
    if any(k in ql for k in _GRAPH_ROUTE_KEYWORDS):
        return "graph", "Relational/cross-document keywords detected; using graph traversal.", "rule"
    if _extract_id_like_tokens(q):
        return "graph", "Identifier tokens detected; using graph traversal for entity links.", "rule"
    return "vector", "No relational signals; using semantic search.", "rule"


@tool
def search_hybrid_rag(query: str, doc_ids: str, top_k: int = 10) -> str:
    """Hybrid retrieval: Qdrant semantic search plus additive Neo4j graph hits.

    query: natural-language need; doc_ids: comma-separated; top_k: semantic results.
    Returns JSON with merged results.
    """
    ids = [d.strip() for d in doc_ids.split(",") if d.strip()]

    # 1) Router picks retrieval backend for this query.
    route, route_reason, routed_by = _route_query(query)

    # 2) Execute selected branch.
    vector_results = []
    graph_results = []
    keyword_results: list[dict] = []
    effective_route = route

    # Sparse channel for ID-like tokens dense embeddings miss.
    id_tokens = _extract_id_like_tokens(query)
    for tok in id_tokens:
        keyword_results.extend(
            keyword_search(keyword=tok, doc_ids=ids if ids else None, top_k=max(3, top_k // 2))
        )

    # Semantic search always runs; graph hits are additive, never a replacement.
    vector_results = semantic_search(query=query, doc_ids=ids if ids else None, top_k=top_k)
    if route == "graph":
        if ids:
            query_entities = _extract_query_entities(query)
            if query_entities:
                graph_results = query_graph_for_entities(query_entities, ids, depth=2)
            if not graph_results:
                effective_route = "vector"
                route_reason = "Graph traversal found no linked entities; semantic results only."
        else:
            effective_route = "vector"
            route_reason = "Graph traversal requires doc_ids; semantic results only."

    # 3) Merge and deduplicate
    seen_texts = set()
    merged = []

    # Exact identifier hits first.
    for r in keyword_results:
        key = (r.get("text") or "")[:100]
        if key and key not in seen_texts:
            seen_texts.add(key)
            merged.append({
                "source": "keyword",
                "doc_id": r.get("doc_id", ""),
                "page": r.get("page_num", 0),
                "language": r.get("language", ""),
                "score": 1.0,
                "text": r.get("text") or "",
            })

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
        "keyword_count": len(keyword_results),
        "id_tokens": id_tokens,
        "total": len(merged),
    })
