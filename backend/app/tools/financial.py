"""Financial analysis tools for the Cross-Checker agent."""
import json
import re
from typing import Optional
from langchain_core.tools import tool
from app.services.vector_store import semantic_search
from app.utils.arabic_normalizer import extract_amounts
from app.utils.money_parse import parse_monetary_amount, amounts_within_tiny_tolerance


def _parse_amount(value: str) -> Optional[float]:
    """Parse a monetary value string to float. Delegates to deterministic single-token parser."""
    return parse_monetary_amount(value)


@tool
def extract_financial_figures(doc_id: str, figure_type: str = "all") -> str:
    """
    Extract all financial figures (amounts, dates, parties) from a specific document.

    Args:
        doc_id: The document ID to extract from
        figure_type: Type of figures to extract: "amounts", "dates", "parties", or "all"

    Returns:
        JSON string with all extracted financial figures from the document
    """
    # Query Qdrant for chunks from this document
    results = semantic_search(
        query="amount total sum payment contract value date party name",
        doc_ids=[doc_id],
        top_k=20,
    )

    figures: dict = {"amounts": [], "dates": [], "parties": [], "raw_passages": []}

    # Patterns
    date_pattern = re.compile(
        r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}|'
        r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|'
        r'يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\s+\d{4})\b',
        re.IGNORECASE | re.UNICODE
    )

    seen_amounts = set()
    seen_dates = set()

    for chunk in results:
        text = chunk.get("text", "")
        page = chunk.get("page_num", 0)

        if figure_type in ("amounts", "all"):
            for amt in extract_amounts(text):
                key = amt["raw"]
                if key not in seen_amounts:
                    seen_amounts.add(key)
                    parsed = _parse_amount(key)
                    figures["amounts"].append({
                        "raw": key,
                        "parsed_value": parsed,
                        "page": page,
                        "doc_id": doc_id,
                    })

        if figure_type in ("dates", "all"):
            for m in date_pattern.finditer(text):
                date_str = m.group(0)
                if date_str not in seen_dates:
                    seen_dates.add(date_str)
                    figures["dates"].append({"date": date_str, "page": page})

        figures["raw_passages"].append({"page": page, "text": text[:300]})

    return json.dumps({
        "doc_id": doc_id,
        "figures": figures,
        "amount_count": len(figures["amounts"]),
        "date_count": len(figures["dates"]),
    })


@tool
def compare_values(value1: str, value2: str, context: str = "") -> str:
    """
    Compare two financial values and determine if they represent a contradiction.
    Assigns a severity score and explanation.

    Args:
        value1: First monetary value only (no filenames or doc labels — use context for provenance)
        value2: Second monetary value only
        context: Additional context (documents, relationship, evidence references)

    Returns:
        JSON string with contradiction assessment, severity, and confidence score
    """
    num1 = _parse_amount(value1)
    num2 = _parse_amount(value2)

    result: dict = {
        "value1": value1,
        "value2": value2,
        "context": context,
        "is_contradiction": False,
        "severity": "ok",
        "confidence": 0.0,
        "discrepancy_percent": None,
        "explanation": "",
    }

    if num1 is None or num2 is None:
        result["is_contradiction"] = False
        result["severity"] = "ok"
        result["confidence"] = 0.4
        result["explanation"] = (
            "Could not parse one or both values as unambiguous monetary amounts; "
            "skipping numeric contradiction. " + (context or "")
        )
        return json.dumps(result)

    if amounts_within_tiny_tolerance(num1, num2):
        result["is_contradiction"] = False
        result["severity"] = "ok"
        result["confidence"] = 0.95
        result["explanation"] = f"Values match within tolerance: {num1:,.6g} ≈ {num2:,.6g}"
        return json.dumps(result)

    diff = abs(num1 - num2)
    avg = (num1 + num2) / 2
    pct = (diff / avg * 100) if avg > 0 else 0
    result["discrepancy_percent"] = round(pct, 4)

    # Sub-percent drift: treat as consistent (formatting / rounding)
    if pct < 1.0 and diff <= max(0.01 * max(abs(num1), abs(num2)), 1.0):
        result["is_contradiction"] = False
        result["severity"] = "ok"
        result["confidence"] = 0.9
        result["explanation"] = (
            f"Values agree within rounding tolerance ({pct:.3f}% difference). "
            f"{num1:,.6g} vs {num2:,.6g}. {context}"
        )
        return json.dumps(result)

    result["is_contradiction"] = True

    if pct >= 10:
        result["severity"] = "critical"
        result["confidence"] = 0.95
        result["explanation"] = (
            f"CRITICAL DISCREPANCY: {pct:.1f}% difference. "
            f"Value 1: {num1:,.2f} vs Value 2: {num2:,.2f}. "
            f"Difference: {diff:,.2f}. {context}"
        )
    elif pct >= 2:
        result["severity"] = "warning"
        result["confidence"] = 0.85
        result["explanation"] = (
            f"WARNING: {pct:.1f}% difference detected. "
            f"Value 1: {num1:,.2f} vs Value 2: {num2:,.2f}. "
            f"Possible rounding or versioning issue. {context}"
        )
    else:
        result["severity"] = "warning"
        result["confidence"] = 0.6
        result["explanation"] = (
            f"Minor difference ({pct:.2f}%) — likely rounding. "
            f"Value 1: {num1:,.2f} vs Value 2: {num2:,.2f}."
        )

    return json.dumps(result)
