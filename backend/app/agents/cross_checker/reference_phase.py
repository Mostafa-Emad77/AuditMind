"""Phase 1c: deterministic cross-document self-reference validation."""
import logging

from app.models.schemas import Finding
from app.agents.cross_checker.context import CrossCheckContext
from app.agents.cross_checker.evidence import _emit

logger = logging.getLogger(__name__)


async def run_reference_phase(
    ctx: CrossCheckContext,
    *,
    find_fn,
    accept_fn,
) -> list[Finding]:
    """A value one document claims about another vs. that document's own figure.

    Deterministic: a disagreeing restatement IS a contradiction — emitted directly.
    """
    findings: list[Finding] = []
    writer = ctx.writer

    try:
        reference_rows = await ctx.loop.run_in_executor(
            None, find_fn, ctx.doc_ids
        )
        if reference_rows:
            step = _emit(writer, "tool_result",
                         f"Reference validation: {len(reference_rows)} restated value(s) "
                         "disagree with the document they cite.",
                         tool_name="find_reference_mismatches",
                         tool_output=f"{len(reference_rows)} mismatches")
            ctx.new_steps.append(step)
        for row in reference_rows:
            claim_doc = row.get("claim_doc_name") or "Document"
            truth_doc = row.get("truth_doc_name") or "Referenced document"
            anchor = str(row.get("anchor_value") or "").strip()
            label = row.get("label", "referenced value")
            claim_raw = str(row.get("claim_raw") or row.get("claim_value"))
            truth_raw = str(row.get("truth_raw") or row.get("truth_value"))
            cv = float(row["claim_value"])
            tv = float(row["truth_value"])
            rel = float(row.get("relative_difference") or 0.0)
            severity = "critical" if (rel >= 0.05 or abs(cv - tv) >= 5000) else "warning"
            candidate = Finding(
                severity=severity,
                title=(
                    f"{claim_doc} restates {label} of {anchor or truth_doc} as {claim_raw}; "
                    f"{truth_doc} states {truth_raw}"
                ),
                description=(
                    f"{claim_doc} (page {row.get('claim_page', 0)}) cites {anchor or truth_doc} "
                    f"with a {label} of {claim_raw}. {truth_doc} itself (page "
                    f"{row.get('truth_page', 0)}) states {truth_raw}. Difference: "
                    f"{cv - tv:+,.2f} ({rel:.1%}). A document's reference to another "
                    f"document must match that document's own stated value."
                ),
                confidence_score=0.95,
                source_doc_id=row.get("claim_doc_id"),
                source_page=row.get("claim_page", 0),
                conflicting_doc_id=row.get("truth_doc_id"),
                conflicting_page=row.get("truth_page", 0),
                evidence=[
                    f"{claim_doc}, page {row.get('claim_page', 0)}: {claim_raw}",
                    f"{truth_doc}, page {row.get('truth_page', 0)}: {truth_raw}",
                ],
                recommendation=(
                    f"Confirm which {label} is correct with the issuing parties and have "
                    f"{claim_doc} reissued to cite {truth_doc}'s stated figure."
                ),
            )
            ctx.candidates_total += 1
            accepted, rejection_reason = accept_fn(
                candidate, source="detector", **ctx.gate_kwargs
            )
            if accepted and ctx.is_suppressed(candidate):
                step = _emit(writer, "thought",
                             f"Skipping reference finding suppressed by prior feedback: {claim_raw} vs {truth_raw}")
                ctx.new_steps.append(step)
                continue
            if accepted:
                ctx.accepted_total += 1
                findings.append(candidate)
                step = _emit(writer, "finding",
                             f"REFERENCE MISMATCH: {claim_doc} says {label} = {claim_raw}; "
                             f"{truth_doc} says {truth_raw}\n"
                             f"Severity: {severity.upper()} | Confidence: 95%")
                ctx.new_steps.append(step)
            else:
                logger.info("Gate rejected reference finding %r: %s",
                            candidate.title[:60], rejection_reason)
    except Exception as e:
        logger.warning("Reference validation failed: %s", e, exc_info=True)

    return findings
