"""Tests for list_sessions() — the Redis scan backing the Archive page's /api/audits."""
import asyncio

import app.services.redis_store as redis_store
from app.models.schemas import AuditSession, DocumentMeta


class _FakeRedis:
    """Minimal stand-in for the two Redis calls list_sessions() makes."""

    def __init__(self, sessions: dict[str, AuditSession]):
        self._raw = {f"session:{k}": v.model_dump_json() for k, v in sessions.items()}

    async def scan_iter(self, pattern: str):
        for key in self._raw:
            yield key

    async def get(self, key: str):
        return self._raw.get(key)


def _session(audit_id: str, api_key: str, created_offset_minutes: int = 0) -> AuditSession:
    import datetime as dt
    return AuditSession(
        audit_id=audit_id,
        api_key=api_key,
        documents=[DocumentMeta(filename=f"{audit_id}.pdf")],
        status="completed",
        created_at=dt.datetime(2026, 1, 1) + dt.timedelta(minutes=created_offset_minutes),
    )


class TestListSessionsScoping:
    def test_only_returns_sessions_for_the_caller_api_key(self, monkeypatch):
        sessions = {
            "a1": _session("a1", api_key="key-alice"),
            "a2": _session("a2", api_key="key-bob"),
            "a3": _session("a3", api_key="key-alice"),
        }
        fake = _FakeRedis(sessions)
        monkeypatch.setattr(redis_store, "get_redis", lambda: _async_return(fake))

        result = asyncio.run(redis_store.list_sessions("key-alice"))

        assert {s.audit_id for s in result} == {"a1", "a3"}

    def test_default_scope_excludes_other_keys(self, monkeypatch):
        sessions = {
            "a1": _session("a1", api_key="default"),
            "a2": _session("a2", api_key="key-bob"),
        }
        fake = _FakeRedis(sessions)
        monkeypatch.setattr(redis_store, "get_redis", lambda: _async_return(fake))

        result = asyncio.run(redis_store.list_sessions("default"))

        assert [s.audit_id for s in result] == ["a1"]

    def test_sorted_newest_first(self, monkeypatch):
        sessions = {
            "oldest": _session("oldest", api_key="k", created_offset_minutes=0),
            "newest": _session("newest", api_key="k", created_offset_minutes=10),
            "middle": _session("middle", api_key="k", created_offset_minutes=5),
        }
        fake = _FakeRedis(sessions)
        monkeypatch.setattr(redis_store, "get_redis", lambda: _async_return(fake))

        result = asyncio.run(redis_store.list_sessions("k"))

        assert [s.audit_id for s in result] == ["newest", "middle", "oldest"]

    def test_empty_scope_returns_empty_list(self, monkeypatch):
        fake = _FakeRedis({})
        monkeypatch.setattr(redis_store, "get_redis", lambda: _async_return(fake))

        assert asyncio.run(redis_store.list_sessions("k")) == []


async def _async_return(value):
    return value
