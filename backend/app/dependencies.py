"""Shared FastAPI dependencies."""
from fastapi import Header, HTTPException, Query

from app.config import get_settings


async def require_api_key(
    x_api_key: str = Header(default=""),
    api_key: str = Query(default=""),
) -> str:
    """Validate the API key (query param allowed: EventSource can't set headers).
    With API_KEYS unset, auth is off and callers share the "default" scope."""
    settings = get_settings()
    keys = settings.api_keys_list
    if not keys:
        return "default"

    resolved = x_api_key or api_key
    if resolved not in keys:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
    return resolved
