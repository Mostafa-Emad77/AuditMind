"""
Redis-backed session and report store.
Replaces the in-memory _sessions / _reports dicts in main.py.

Keys used:
  session:{audit_id}  → JSON-serialized AuditSession   (TTL: 7 days)
  report:{audit_id}   → JSON-serialized AuditReport    (TTL: 7 days)
"""
import json
import logging
from typing import Optional

import redis.asyncio as aioredis

from app.config import get_settings
from app.models.schemas import AuditReport, AuditSession

logger = logging.getLogger(__name__)

# TTL for all keys: 7 days (fits comfortably inside the 30 MB free tier
# because expired keys are evicted automatically)
_TTL_SECONDS = 60 * 60 * 24 * 7

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """Return a singleton async Redis client, creating it on first call."""
    global _redis
    if _redis is None:
        settings = get_settings()
        _redis = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,       # always return str, not bytes
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        # Verify connectivity on startup
        await _redis.ping()
        logger.info("Redis connected: %s", settings.redis_url.split("@")[-1])
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


# ─── Session helpers ──────────────────────────────────────────────────────────

async def save_session(session: AuditSession) -> None:
    r = await get_redis()
    await r.set(
        f"session:{session.audit_id}",
        session.model_dump_json(),
        ex=_TTL_SECONDS,
    )


async def get_session(audit_id: str) -> Optional[AuditSession]:
    r = await get_redis()
    raw = await r.get(f"session:{audit_id}")
    if raw is None:
        return None
    return AuditSession.model_validate_json(raw)


async def delete_session(audit_id: str) -> None:
    r = await get_redis()
    await r.delete(f"session:{audit_id}")


# ─── Report helpers ───────────────────────────────────────────────────────────

async def save_report(report: AuditReport) -> None:
    r = await get_redis()
    await r.set(
        f"report:{report.audit_id}",
        report.model_dump_json(),
        ex=_TTL_SECONDS,
    )


async def get_report(audit_id: str) -> Optional[AuditReport]:
    r = await get_redis()
    raw = await r.get(f"report:{audit_id}")
    if raw is None:
        return None
    return AuditReport.model_validate_json(raw)


# ─── Document lookup helper ───────────────────────────────────────────────────

async def find_document(doc_id: str):
    """Scan all sessions to find a document by doc_id.
    Only used by the /documents/{doc_id} endpoint — not a hot path.
    """
    r = await get_redis()
    # Use SCAN to avoid blocking KEYS in production
    async for key in r.scan_iter("session:*"):
        raw = await r.get(key)
        if not raw:
            continue
        session = AuditSession.model_validate_json(raw)
        for doc in session.documents:
            if doc.doc_id == doc_id:
                return doc
    return None
