"""Cross-Checker Agent package — the core contradiction detection engine.

Split (Phase 2.5) from the former 1,062-line ``cross_checker.py`` module into:
  - adjudication.py — LLM calls that assess/adjudicate candidate contradictions
  - gates.py         — the precision-first finding acceptance gate
  - dedup.py         — topic bucketing, numeric/ref signatures, the dedup loop helpers
  - agent.py         — orchestration (the ``cross_checker_agent`` LangGraph node)

This ``__init__`` re-exports the public entry point plus the pure helper
functions that are unit-tested directly (see ``tests/test_cross_checker_regression.py``),
so ``from app.agents.cross_checker import ...`` keeps working unchanged.
"""
from app.agents.cross_checker.agent import cross_checker_agent
from app.agents.cross_checker.evidence import (
    _emit,
    _evidence_line_backs_value,
    _evidence_snippets_back_values,
    _graph_context_for_compare,
    _is_bank_doc,
    _classify_check_intents,
)
from app.agents.cross_checker.gates import _accept_finding
from app.agents.cross_checker.dedup import (
    _finding_topic_bucket,
    _finding_ref_signature,
    _numeric_signature_overlap,
    _finding_numeric_signature,
    _prefer_finding,
    _topics_compatible_for_dedup,
)
from app.agents.cross_checker.adjudication import (
    _llm_assess_amount_pair,
    _llm_adjudicate_check,
)

__all__ = [
    "cross_checker_agent",
    "_emit",
    "_evidence_line_backs_value",
    "_evidence_snippets_back_values",
    "_graph_context_for_compare",
    "_is_bank_doc",
    "_classify_check_intents",
    "_accept_finding",
    "_finding_topic_bucket",
    "_finding_ref_signature",
    "_numeric_signature_overlap",
    "_finding_numeric_signature",
    "_prefer_finding",
    "_topics_compatible_for_dedup",
    "_llm_assess_amount_pair",
    "_llm_adjudicate_check",
]
