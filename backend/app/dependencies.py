"""Shared FastAPI dependencies."""
from fastapi import Header, HTTPException, Query

from app.config import get_settings


async def require_api_key(
    x_api_key: str = Header(default=""),
    api_key: str = Query(default=""),
) -> str:
    """
    Validate the caller's API key against the configured allowlist.

    If API_KEYS is unset, auth is disabled and every caller resolves to the
    shared "default" scope (today's behavior, unchanged). ``api_key`` is
    accepted as a query param too because ``EventSource`` (used for the SSE
    stream) cannot set custom headers.
    """
    settings = get_settings()
    keys = settings.api_keys_list
    if not keys:
        return "default"

    resolved = x_api_key or api_key
    if resolved not in keys:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
    return resolved
