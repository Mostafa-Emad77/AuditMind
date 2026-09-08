"""Post-generation guards on findings. Internal ids are UUIDs; one cited in a
finding but absent from its evidence is a hallucinated reference."""
import logging
import re

from app.models.schemas import Finding

logger = logging.getLogger(__name__)

# 8-4-4-4-12 hex — the canonical UUID shape used for every internal id in this app.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def _finding_prose(f: Finding) -> str:
    return " ".join(
        part for part in (f.title, f.description or "", f.recommendation or "") if part
    )


def _finding_evidence_text(f: Finding) -> str:
    return " ".join(e for e in (f.evidence or []) if e)


def drop_hallucinated_identifier_findings(
    findings: list[Finding],
) -> tuple[list[Finding], int]:
    """Drop findings citing a UUID absent from their evidence → (kept, dropped_count)."""
    kept: list[Finding] = []
    dropped = 0
    for f in findings:
        prose_uuids = set(_UUID_RE.findall(_finding_prose(f)))
        if not prose_uuids:
            kept.append(f)
            continue
        evidence_text = _finding_evidence_text(f)
        unbacked = {u for u in prose_uuids if u.lower() not in evidence_text.lower()}
        if unbacked:
            dropped += 1
            logger.warning(
                "Discarding finding %r: cites identifier(s) %s not present in its "
                "evidence — treating as hallucinated internal id.",
                f.title[:80],
                ", ".join(sorted(unbacked)),
            )
            continue
        kept.append(f)
    return kept, dropped
