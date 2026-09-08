"""Phase 3: dedup + hallucinated-identifier guard + summary."""
import logging

from app.models.schemas import Finding
from app.agents.cross_checker.context import CrossCheckContext
from app.agents.cross_checker.dedup import dedupe_findings
from app.agents.cross_checker.evidence import _emit
from app.agents.cross_checker.validation import drop_hallucinated_identifier_findings

logger = logging.getLogger(__name__)


def finalize_findings(ctx: CrossCheckContext, findings: list[Finding]) -> list[Finding]:
    """Deduplicate findings and drop ones citing unverifiable UUID identifiers."""
    logger.info(
        "Precision gate summary: %d candidates → %d accepted (%.0f%% pass rate)",
        ctx.candidates_total,
        ctx.accepted_total,
        (ctx.accepted_total / ctx.candidates_total * 100) if ctx.candidates_total else 0.0,
    )

    deduped = dedupe_findings(findings)
    if len(deduped) < len(findings):
        logger.info("Deduplicated findings: %d → %d", len(findings), len(deduped))
    findings = deduped

    findings, _hallucinated = drop_hallucinated_identifier_findings(findings)
    if _hallucinated:
        logger.warning(
            "Dropped %d finding(s) citing unverifiable UUID-format identifiers", _hallucinated,
        )
        step = _emit(ctx.writer, "thought",
                     f"Discarded {_hallucinated} finding(s) that referenced an internal "
                     "identifier not present in any source document.")
        ctx.new_steps.append(step)

    return findings


def build_summary(ctx: CrossCheckContext, findings: list[Finding]) -> str:
    critical = sum(1 for f in findings if f.severity == "critical")
    warnings = sum(1 for f in findings if f.severity == "warning")
    ok_count = sum(1 for f in findings if f.severity == "ok")

    summary = (
        f"Cross-checking complete. "
        f"Results: {critical} critical issue(s), {warnings} warning(s), {ok_count} OK. "
        f"Total findings: {len(findings)}."
    )
    if critical > 0:
        summary += " CRITICAL issues require immediate attention."
    elif warnings > 0:
        summary += " Warnings should be reviewed before finalizing."
    else:
        summary += " Documents appear largely consistent."

    step = _emit(ctx.writer, "summary", summary)
    ctx.new_steps.append(step)
    return summary
