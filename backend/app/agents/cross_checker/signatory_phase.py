"""Phase 1b: signatory authority mismatches.

Narrow by design: only an amendment/successor signed BELOW the original signer's
level. A QS → engineer → PM approval chain is normal, not a mismatch.
"""
import logging

from app.models.schemas import Finding
from app.agents.cross_checker.context import CrossCheckContext
from app.agents.cross_checker.evidence import _emit

logger = logging.getLogger(__name__)


async def run_signatory_phase(
    ctx: CrossCheckContext,
    *,
    find_fn,
    accept_fn,
) -> list[Finding]:
    findings: list[Finding] = []
    writer = ctx.writer

    try:
        signatory_rows = await ctx.loop.run_in_executor(
            None, find_fn, ctx.doc_ids
        )
        for row in signatory_rows:
            n1 = str(row.get("name1") or "").strip()
            n2 = str(row.get("name2") or "").strip()
            t1 = str(row.get("title1") or row.get("authority1") or "unknown").strip()
            t2 = str(row.get("title2") or row.get("authority2") or "unknown").strip()
            if not n1 or not n2:
                continue
            doc1 = row.get("doc1_name") or row.get("doc1_id") or "Doc 1"
            doc2 = row.get("doc2_name") or row.get("doc2_id") or "Doc 2"
            signed1 = ", ".join(row.get("signed1") or []) or "unspecified"
            signed2 = ", ".join(row.get("signed2") or []) or "unspecified"
            gap = int(row.get("authority_gap") or 0)

            candidate = Finding(
                severity="warning",
                title=f"Amendment signed below original authority: {n2} ({t2}) vs {n1} ({t1})",
                description=(
                    f"{signed1} was signed by {n1} ({t1}); its stated amendment/successor "
                    f"{signed2} was signed by {n2} ({t2}), a lower authority level. An "
                    f"amendment should be approved at or above the authority level of the "
                    f"original agreement unless delegation of that authority is documented."
                ),
                confidence_score=0.7 if gap >= 2 else 0.6,
                source_doc_id=row.get("doc1_id"),
                source_page=row.get("page1", 0),
                conflicting_doc_id=row.get("doc2_id"),
                conflicting_page=row.get("page2", 0),
                evidence=[
                    f"{doc1}: {n1} — {t1} (signed: {signed1})",
                    f"{doc2}: {n2} — {t2} (signed: {signed2})",
                ],
                recommendation=(
                    "Confirm the signatory held delegated authority for the value and "
                    "scope of the document they executed, and that any amendment was "
                    "approved at or above the authority level of the original agreement."
                ),
            )
            ctx.candidates_total += 1
            accepted, rejection_reason = accept_fn(
                candidate, source="graph", **ctx.gate_kwargs
            )
            if accepted and ctx.is_suppressed(candidate):
                step = _emit(writer, "thought",
                             f"Skipping signatory finding suppressed by prior feedback: {n1} vs {n2}")
                ctx.new_steps.append(step)
                continue
            if accepted:
                ctx.accepted_total += 1
                findings.append(candidate)
                step = _emit(writer, "finding",
                             f"AMENDMENT AUTHORITY MISMATCH: {n2} ({t2}) countersigned "
                             f"{signed2}; original {signed1} signed by {n1} ({t1})")
                ctx.new_steps.append(step)
            else:
                logger.info("Gate rejected signatory finding %r: %s",
                            candidate.title[:60], rejection_reason)
    except Exception as e:
        logger.warning("Signatory mismatch detection failed: %s", e, exc_info=True)

    return findings
