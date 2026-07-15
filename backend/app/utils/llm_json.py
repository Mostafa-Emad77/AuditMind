"""Shared helpers for parsing JSON out of LLM responses.

Consolidates the brace-balanced JSON extractor and the "flatten list content"
LLM-response normalizer that were previously duplicated across cross_checker.py,
document_processor.py, planner_tools.py, and hybrid_retriever.py.
"""


def extract_json_obj(text: str) -> dict | None:
    """Extract the first JSON object from model output (brace-balanced scan, no regex)."""
    if not text:
        return None
    import json

    t = text.strip()
    try:
        parsed = json.loads(t)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    quote = ""
    i = start
    while i < len(t):
        c = t[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == quote:
                in_str = False
            i += 1
            continue
        if c in "\"'":
            in_str = True
            quote = c
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = t[start : i + 1]
                try:
                    parsed = json.loads(chunk)
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
        i += 1
    return None


def normalize_llm_content(resp) -> str:
    """Flatten a LangChain LLM response's `.content` into a plain string.

    Handles the common case where `.content` is a list of content blocks
    (e.g. some providers return `[{"type": "text", "text": "..."}]`).
    """
    content = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(content, list):
        content = "".join(
            c.get("text", "") if isinstance(c, dict) else str(c)
            for c in content
        )
    return str(content)
