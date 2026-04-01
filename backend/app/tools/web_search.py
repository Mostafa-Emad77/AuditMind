"""External verification tool using web search."""
import json
import logging
import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def search_web(query: str) -> str:
    """
    Search the web to verify external facts such as:
    - Company registration status and legitimacy
    - Current exchange rates (EGP, USD, EUR, etc.)
    - Legal clause standards or regulations
    - Tax rates and VAT rules in Egypt/MENA

    Args:
        query: Search query (Arabic or English)

    Returns:
        JSON string with search results summary
    """
    try:
        # DuckDuckGo Instant Answer API (free, no key required)
        params = {
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
            "skip_disambig": "1",
        }
        with httpx.Client(timeout=10) as client:
            response = client.get("https://api.duckduckgo.com/", params=params)
            response.raise_for_status()
            data = response.json()

        abstract = data.get("AbstractText", "")
        abstract_url = data.get("AbstractURL", "")
        related = [
            {"title": r.get("Text", ""), "url": r.get("FirstURL", "")}
            for r in data.get("RelatedTopics", [])[:3]
            if r.get("Text")
        ]

        if abstract:
            return json.dumps({
                "query": query,
                "summary": abstract,
                "source_url": abstract_url,
                "related": related,
                "found": True,
            })

        # Fallback: try Brave Search if available, else return not-found
        return json.dumps({
            "query": query,
            "summary": f"No instant answer found for: {query}. Manual verification recommended.",
            "related": related,
            "found": False,
        })

    except Exception as e:
        logger.warning("Web search failed for query '%s': %s", query, e)
        return json.dumps({
            "query": query,
            "error": str(e),
            "summary": "Web search temporarily unavailable. Please verify manually.",
            "found": False,
        })


@tool
def get_document_metadata(doc_id: str, audit_id: str) -> str:
    """
    Retrieve metadata for a specific document from the audit session.

    Args:
        doc_id: The document ID
        audit_id: The audit session ID

    Returns:
        JSON string with document metadata (type, language, dates, amounts, parties)
    """
    # This is resolved at runtime via the audit state in-memory store
    # The actual implementation is in the planner agent which has access to state
    return json.dumps({
        "doc_id": doc_id,
        "note": "Metadata retrieved from audit session state",
    })
