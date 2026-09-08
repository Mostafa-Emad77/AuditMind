"""LangGraph state definitions for AuditMind."""
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from app.models.schemas import (
    DocumentMeta,
    ChecklistItem,
    Finding,
    ReasoningStep,
    AuditReport,
)


class AuditState(TypedDict):
    """Shared state passed between all LangGraph agent nodes."""

    # Conversation messages (LangGraph managed)
    messages: Annotated[list, add_messages]

    # Audit session
    audit_id: str

    # Uploaded documents metadata
    documents: list[DocumentMeta]

    # Planner output
    checklist: list[ChecklistItem]

    # Re-plan loop counter (planner increments each pass; router caps total passes)
    replan_count: Annotated[int, lambda a, b: a + b]

    # Cross-Checker output
    findings: list[Finding]

    # Report Writer output
    report: Optional[AuditReport]

    # Live reasoning trace (streamed to UI)
    reasoning_trace: Annotated[list[ReasoningStep], lambda a, b: a + b]

    # Report language preference
    report_language: str  # "arabic" | "english"

    # Error tracking
    error: Optional[str]

    # Scope for the suppressed-findings feedback loop (see AuditSession.api_key)
    api_key: str
