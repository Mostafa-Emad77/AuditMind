"""
Redis-backed session and report store.
Replaces the in-memory _sessions / _reports dicts in main.py.

Keys used:
  session:{audit_id}          → JSON-serialized AuditSession   (TTL: 7 days)
  report:{audit_id}           → JSON-serialized AuditReport    (TTL: 7 days)
  chat:{audit_id}              → JSON list of {role, content}   (TTL: 7 days)
  triage:{audit_id}           → JSON map finding_id → record   (TTL: 7 days)
  events:{audit_id}           → List of persisted SSE events   (TTL: 7 days)
  heartbeat:{audit_id}        → "1"                             (TTL: 2 min — liveness signal)
  suppressed:{scope}:signatures → Set of finding signatures    (no TTL — per-API-key feedback)
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


async def delete_report(audit_id: str) -> None:
    r = await get_redis()
    await r.delete(f"report:{audit_id}")


async def purge_audit(audit_id: str) -> None:
    """Delete all Redis keys associated with an audit (session, report, chat, triage, events, heartbeat)."""
    r = await get_redis()
    await r.delete(
        f"session:{audit_id}",
        f"report:{audit_id}",
        f"chat:{audit_id}",
        f"triage:{audit_id}",
        f"events:{audit_id}",
        f"heartbeat:{audit_id}",
    )


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


# ─── Chat history helpers ─────────────────────────────────────────────────────

_CHAT_MAX_MESSAGES = 20  # keep the last N turns for context


async def get_chat_history(audit_id: str) -> list[dict]:
    """Return the stored chat history (list of {"role", "content"}) for an audit."""
    r = await get_redis()
    raw = await r.get(f"chat:{audit_id}")
    if not raw:
        return []
    try:
        history = json.loads(raw)
        return history if isinstance(history, list) else []
    except json.JSONDecodeError:
        return []


async def append_chat_messages(audit_id: str, messages: list[dict]) -> None:
    """Append messages to the audit's chat history, trimming to the last N."""
    history = await get_chat_history(audit_id)
    history.extend(messages)
    history = history[-_CHAT_MAX_MESSAGES:]
    r = await get_redis()
    await r.set(f"chat:{audit_id}", json.dumps(history, ensure_ascii=False), ex=_TTL_SECONDS)


# ─── Finding triage helpers ───────────────────────────────────────────────────

async def get_triage(audit_id: str) -> dict[str, dict]:
    """Return the triage map (finding_id → {status, note, signature, updated_at})."""
    r = await get_redis()
    raw = await r.get(f"triage:{audit_id}")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


async def set_triage(audit_id: str, finding_id: str, record: dict) -> dict[str, dict]:
    """Upsert a single finding's triage record; returns the full updated map."""
    triage = await get_triage(audit_id)
    triage[finding_id] = record
    r = await get_redis()
    await r.set(f"triage:{audit_id}", json.dumps(triage, ensure_ascii=False), ex=_TTL_SECONDS)
    return triage


# ─── Suppressed-signature feedback loop ───────────────────────────────────────
# Scoped per API key (``scope`` — "default" when auth is disabled) so one caller's
# false-positive dismissals don't silently suppress another caller's findings.


def _suppressed_key(scope: str) -> str:
    return f"suppressed:{scope}:signatures"


async def add_suppressed_signature(scope: str, signature: str) -> None:
    """Record a finding signature dismissed as a false positive (no TTL)."""
    if not signature:
        return
    r = await get_redis()
    await r.sadd(_suppressed_key(scope), signature)


async def remove_suppressed_signature(scope: str, signature: str) -> None:
    """Undo a suppression (e.g. when a finding's triage status changes back)."""
    if not signature:
        return
    r = await get_redis()
    await r.srem(_suppressed_key(scope), signature)


async def get_suppressed_signatures(scope: str) -> set[str]:
    """Return the set of suppressed finding signatures for a scope."""
    r = await get_redis()
    members = await r.smembers(_suppressed_key(scope))
    return set(members) if members else set()


# ─── Audit heartbeat (liveness signal for stale-session recovery) ────────────

_HEARTBEAT_TTL_SECONDS = 120


async def touch_heartbeat(audit_id: str) -> None:
    """Refresh the self-expiring liveness key for a running audit."""
    r = await get_redis()
    await r.set(f"heartbeat:{audit_id}", "1", ex=_HEARTBEAT_TTL_SECONDS)


async def is_heartbeat_alive(audit_id: str) -> bool:
    """True if the audit's pipeline touched its heartbeat within the last TTL window."""
    r = await get_redis()
    return bool(await r.exists(f"heartbeat:{audit_id}"))


# ─── Audit event log (background execution decoupled from the SSE connection) ─
# The pipeline appends every SSE-shaped event (reasoning_step, report_ready, complete,
# error) to a Redis list as it runs. The `/stream` endpoint is a pure read-only
# subscriber: it replays this list from the beginning, then tails new entries as they
# arrive — so a client disconnecting (tab close, network blip) no longer kills the
# audit, and reconnecting resumes the trace instead of losing it.

async def append_audit_event(audit_id: str, event: dict) -> None:
    """Append one SSE-shaped event to the audit's persisted event log."""
    r = await get_redis()
    await r.rpush(f"events:{audit_id}", json.dumps(event, ensure_ascii=False, default=str))
    await r.expire(f"events:{audit_id}", _TTL_SECONDS)


async def get_audit_events(audit_id: str, start: int = 0) -> list[dict]:
    """Return persisted events for an audit from index `start` onward."""
    r = await get_redis()
    raw = await r.lrange(f"events:{audit_id}", start, -1)
    return [json.loads(item) for item in raw]


async def get_audit_events_count(audit_id: str) -> int:
    r = await get_redis()
    return await r.llen(f"events:{audit_id}")


async def clear_audit_events(audit_id: str) -> None:
    """Drop a prior run's event log before restarting a stale/failed audit."""
    r = await get_redis()
    await r.delete(f"events:{audit_id}")


# ─── Document lookup helper ───────────────────────────────────────────────────

async def list_sessions(api_key: str) -> list[AuditSession]:
    """Return every session in one API-key scope, newest first — powers the Archive list.

    Full SCAN, same acceptable-at-current-scale tradeoff as find_document() (see
    roadmap D4) — revisit with a secondary index if audit volume grows.
    """
    r = await get_redis()
    sessions: list[AuditSession] = []
    async for key in r.scan_iter("session:*"):
        raw = await r.get(key)
        if not raw:
            continue
        session = AuditSession.model_validate_json(raw)
        if session.api_key == api_key:
            sessions.append(session)
    sessions.sort(key=lambda s: s.created_at, reverse=True)
    return sessions


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
