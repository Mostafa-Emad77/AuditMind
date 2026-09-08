"""Precision-first acceptance gate for all cross-checker finding paths."""
from app.models.schemas import Finding


def _accept_finding(
    finding: Finding,
    *,
    source: str = "llm",
    min_evidence_for_critical: int = 2,
    min_confidence_llm: float = 0.65,
    min_confidence_graph: float = 0.55,
) -> tuple[bool, str]:
    """Precision gate for every finding path. source: llm | graph | detector.
    Returns (accepted, rejection_reason)."""
    # ok-severity findings are observational — always let through
    if finding.severity == "ok":
        return True, ""

    # Confidence floor per source
    if source == "llm":
        min_conf = min_confidence_llm
    elif source == "graph":
        min_conf = min_confidence_graph
    else:
        # rule-based / detector findings are trusted; only a minimal floor
        min_conf = 0.40

    if finding.confidence_score < min_conf:
        return False, (
            f"low_confidence(score={finding.confidence_score:.2f} < floor={min_conf:.2f}, source={source})"
        )

    # Critical LLM findings need corroboration from at least 2 distinct evidence snippets
    if source == "llm" and finding.severity == "critical":
        distinct_ev = len([e for e in finding.evidence if e.strip()])
        if distinct_ev < min_evidence_for_critical:
            return False, (
                f"insufficient_evidence(count={distinct_ev} < required={min_evidence_for_critical}, critical)"
            )

    # Graph findings that are critical must still pass the evidence-backs-value check
    # (already done inline for graph phase, so this is a belt-and-suspenders check)
    if source == "graph" and finding.severity == "critical":
        distinct_ev = len([e for e in finding.evidence if e.strip()])
        if distinct_ev < 2:
            return False, (
                f"insufficient_graph_evidence(count={distinct_ev} < 2, critical)"
            )

    return True, ""
