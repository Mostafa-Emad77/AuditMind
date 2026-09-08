"""Shared context for cross-checker phases — no heavy service imports."""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from app.models.schemas import Finding, ReasoningStep

# Parallel LLM adjudication; kept small — OpenRouter rate-limits per key.
_ADJUDICATION_CONCURRENCY = 3


class _StepBuffer:
    """Buffers a concurrent worker's steps so the trace can be flushed in checklist order."""

    def __init__(self) -> None:
        self._pending: list[tuple[str, str, dict]] = []

    def emit(self, step_type: str, content: str, **kwargs) -> None:
        self._pending.append((step_type, content, kwargs))

    def flush(self, writer, new_steps: list[ReasoningStep]) -> None:
        from app.agents.cross_checker.evidence import _emit

        for step_type, content, kwargs in self._pending:
            new_steps.append(_emit(writer, step_type, content, **kwargs))


@dataclass
class CrossCheckContext:
    """Mutable per-audit state threaded through each phase."""

    writer: Any
    loop: asyncio.AbstractEventLoop
    documents: list
    doc_ids: list[str]
    checklist: list = field(default_factory=list)
    gate_kwargs: dict = field(default_factory=dict)
    new_steps: list[ReasoningStep] = field(default_factory=list)
    candidates_total: int = 0
    accepted_total: int = 0
    is_suppressed: Callable[[Finding], bool] = field(
        default=lambda f: False, repr=False
    )
