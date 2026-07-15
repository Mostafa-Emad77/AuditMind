"""Financial analysis tools for the Cross-Checker agent."""
import json
from typing import Optional
from langchain_core.tools import tool
from app.utils.money_parse import parse_monetary_amount, amounts_within_tiny_tolerance


def _parse_amount(value: str) -> Optional[float]:
    """Parse a monetary value string to float. Delegates to deterministic single-token parser."""
    return parse_monetary_amount(value)


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
