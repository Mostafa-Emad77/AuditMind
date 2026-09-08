"""Executor offload and parallel adjudication must preserve checklist order."""
import asyncio
import json

import pytest

import app.agents.cross_checker.agent as cc
from app.models.schemas import ChecklistItem, DocumentMeta, Finding


def _finding(title: str) -> Finding:
    return Finding(
        severity="warning",
        title=title,
        description="d",
        confidence_score=0.9,
        evidence=["e1", "e2"],
    )


def _checklist(n: int) -> list[ChecklistItem]:
    return [
        ChecklistItem(
            description=f"check-{i}",
            check_type="amount_match",
            doc_ids_involved=["d1", "d2"],
        )
        for i in range(n)
    ]


class _StubTool:
    """Stand-in for a LangChain StructuredTool (a Pydantic model, so not patchable)."""

    def __init__(self, payload: dict):
        self._payload = payload

    def invoke(self, _args) -> str:
        return json.dumps(self._payload)


@pytest.fixture
def wired(monkeypatch):
    """Stub every I/O boundary of cross_checker_agent; return the emitted-step log."""
    emitted: list[dict] = []
    monkeypatch.setattr(cc, "get_stream_writer", lambda: emitted.append)

    monkeypatch.setattr(cc, "detect_graph_contradictions", _StubTool({"contradictions": []}))
    monkeypatch.setattr(
        cc, "search_hybrid_rag",
        _StubTool({"results": [], "vector_count": 0, "graph_count": 0}),
    )
    monkeypatch.setattr(cc, "get_all_chunks_for_docs", lambda _ids: [])
    # Keep the suite hermetic: without this the signatory phase would hit live Neo4j.
    monkeypatch.setattr(cc, "find_signatory_mismatches", lambda _ids: [])

    async def _no_suppression(_key):
        return set()

    monkeypatch.setattr(cc, "get_suppressed_signatures", _no_suppression)
    monkeypatch.setattr(cc, "_accept_finding", lambda f, source, **kw: (True, ""))
    return emitted


def _state(n: int) -> dict:
    return {
        "documents": [
            DocumentMeta(doc_id="d1", filename="a.pdf", doc_type="contract"),
            DocumentMeta(doc_id="d2", filename="b.pdf", doc_type="invoice"),
        ],
        "checklist": _checklist(n),
        "api_key": "default",
    }


class TestChecklistOrdering:
    def test_findings_follow_checklist_order_despite_reverse_completion(
        self, wired, monkeypatch
    ):
        """Later items finish first; the result must still be in checklist order."""
        n = 5

        async def _adjudicate(item, results, documents, max_findings=4):
            idx = int(item.description.split("-")[1])
            # Invert the delay so item 0 resolves last.
            await asyncio.sleep((n - idx) * 0.02)
            return [_finding(f"finding-{idx}")]

        monkeypatch.setattr(cc, "_llm_adjudicate_check", _adjudicate)

        out = asyncio.run(cc.cross_checker_agent(_state(n)))

        titles = [f.title for f in out["findings"]]
        assert titles == [f"finding-{i}" for i in range(n)]

    def test_reasoning_trace_follows_checklist_order(self, wired, monkeypatch):
        n = 4

        async def _adjudicate(item, results, documents, max_findings=4):
            idx = int(item.description.split("-")[1])
            await asyncio.sleep((n - idx) * 0.02)
            return [_finding(f"finding-{idx}")]

        monkeypatch.setattr(cc, "_llm_adjudicate_check", _adjudicate)

        asyncio.run(cc.cross_checker_agent(_state(n)))

        headers = [
            s["step"]["content"] for s in wired
            if s["step"]["content"].startswith("[") and "check-" in s["step"]["content"]
        ]
        assert headers == [f"[{i+1}/{n}] check-{i} [MEDIUM priority]" for i in range(n)]


class TestConcurrency:
    def test_checks_run_in_parallel_bounded_by_the_semaphore(self, wired, monkeypatch):
        in_flight = 0
        peak = 0

        async def _adjudicate(item, results, documents, max_findings=4):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1
            return []

        monkeypatch.setattr(cc, "_llm_adjudicate_check", _adjudicate)

        asyncio.run(cc.cross_checker_agent(_state(8)))

        assert peak > 1, "adjudication should overlap, not run serially"
        assert peak <= cc._ADJUDICATION_CONCURRENCY, "semaphore must bound in-flight calls"


class TestFailureIsolation:
    def test_one_failing_check_does_not_abort_the_others(self, wired, monkeypatch):
        async def _adjudicate(item, results, documents, max_findings=4):
            idx = int(item.description.split("-")[1])
            if idx == 1:
                raise RuntimeError("boom")
            return [_finding(f"finding-{idx}")]

        monkeypatch.setattr(cc, "_llm_adjudicate_check", _adjudicate)

        out = asyncio.run(cc.cross_checker_agent(_state(3)))

        titles = [f.title for f in out["findings"]]
        assert titles == ["finding-0", "finding-2"]
        assert any("Could not complete check" in s["step"]["content"] for s in wired)


class TestGraphPairOrdering:
    def test_pair_findings_follow_query_order(self, wired, monkeypatch):
        """Phase-1 pairs arrive materiality-ordered from Cypher; preserve that."""
        pairs = [
            {
                "norm1": f"{(5 - i) * 1000} EGP", "norm2": f"{(5 - i) * 900} EGP",
                "value1": f"{(5 - i) * 1000} EGP", "value2": f"{(5 - i) * 900} EGP",
                "doc1_id": "d1", "doc2_id": "d2",
                "doc1_name": "a.pdf", "doc2_name": "b.pdf",
                "page1": 1, "page2": 2,
            }
            for i in range(4)
        ]
        monkeypatch.setattr(
            cc, "detect_graph_contradictions", _StubTool({"contradictions": pairs}),
        )

        async def _assess(raw1, raw2, ctx, evidence):
            # Reverse the completion order relative to the input order.
            await asyncio.sleep(float(raw1.split()[0]) / 100000.0)
            return {
                "is_valid_comparison": True,
                "is_contradiction": True,
                "severity": "warning",
                "confidence": 0.9,
                "explanation": "x",
            }

        monkeypatch.setattr(cc, "_llm_assess_amount_pair", _assess)
        monkeypatch.setattr(cc, "_llm_adjudicate_check", None)

        out = asyncio.run(cc.cross_checker_agent({**_state(0)}))

        titles = [f.title for f in out["findings"]]
        expected = [f"Amount Contradiction: {(5 - i) * 1000} EGP vs {(5 - i) * 900} EGP" for i in range(4)]
        assert titles == expected
