"""Re-plan loop: empty findings on a thin plan get one deeper planner pass."""
import asyncio
import json

import app.agents.graph as graph
import app.agents.planner as planner_mod
from app.models.schemas import ChecklistItem, DocumentMeta


def _item(description: str) -> ChecklistItem:
    return ChecklistItem(
        description=description,
        check_type="cross_doc_consistency",
        doc_ids_involved=["d1", "d2"],
        priority="high",
    )


def _docs(n: int) -> list[DocumentMeta]:
    return [
        DocumentMeta(doc_id=f"d{i}", filename=f"{i}.pdf", doc_type="contract", language="english")
        for i in range(n)
    ]


def _state(**over) -> dict:
    base = {"documents": _docs(2), "findings": [], "checklist": [], "replan_count": 1}
    base.update(over)
    return base


class TestReplanRouter:
    def test_loops_back_on_empty_findings_and_thin_plan(self):
        out = graph._route_after_cross_checker(_state(checklist=[object(), object()]))
        assert out == "planner"

    def test_stops_when_findings_exist(self):
        out = graph._route_after_cross_checker(_state(checklist=[], findings=[object()]))
        assert out == "report_writer"

    def test_stops_at_max_passes(self):
        out = graph._route_after_cross_checker(
            _state(checklist=[object()], replan_count=2)
        )
        assert out == "report_writer"

    def test_stops_on_thick_checklist(self):
        out = graph._route_after_cross_checker(_state(checklist=[object()] * 6))
        assert out == "report_writer"

    def test_stops_for_single_doc(self):
        out = graph._route_after_cross_checker(
            _state(documents=_docs(1), checklist=[object()])
        )
        assert out == "report_writer"

    def test_planner_skip_still_first(self):
        assert graph._route_after_planner({"documents": _docs(1)}) == "report_writer"
        assert graph._route_after_planner({"documents": _docs(2)}) == "cross_checker"


class _StubChecklistTool:
    def __init__(self, items=None, boom: bool = False):
        self._items = items or []
        self._boom = boom

    def invoke(self, _args) -> str:
        if self._boom:
            raise RuntimeError("llm down")
        return json.dumps({"checklist": [i.model_dump() for i in self._items]})


def _run_planner(monkeypatch, state: dict, tool) -> dict:
    emitted: list[dict] = []
    monkeypatch.setattr(planner_mod, "get_stream_writer", lambda: emitted.append)
    monkeypatch.setattr(planner_mod, "generate_checklist", tool)
    out = asyncio.run(planner_mod.planner_agent(state))
    out["_emitted"] = emitted
    return out


class TestPlannerReplan:
    def test_first_pass_returns_generated_checklist(self, monkeypatch):
        tool = _StubChecklistTool([_item("Check X"), _item("Check Y")])
        out = _run_planner(monkeypatch, {"documents": _docs(2), "checklist": []}, tool)
        assert [c.description for c in out["checklist"]] == ["Check X", "Check Y"]
        assert out["replan_count"] == 1
        contents = [s["step"]["content"] for s in out["_emitted"]]
        assert not any("Re-plan pass" in c for c in contents)

    def test_replan_dedupes_and_combines(self, monkeypatch):
        prior = [_item("Check X")]
        tool = _StubChecklistTool([_item("Check X"), _item("Check Y")])
        out = _run_planner(
            monkeypatch, {"documents": _docs(2), "checklist": prior}, tool
        )
        assert [c.description for c in out["checklist"]] == ["Check X", "Check Y"]
        assert out["replan_count"] == 1
        contents = [s["step"]["content"] for s in out["_emitted"]]
        assert any("Re-plan pass" in c for c in contents)
        assert any("1 duplicates" in c for c in contents)

    def test_replan_fallback_dedupes(self, monkeypatch):
        prior = [_item("Verify date consistency across all documents")]
        out = _run_planner(
            monkeypatch,
            {"documents": _docs(2), "checklist": prior},
            _StubChecklistTool(boom=True),
        )
        descriptions = [c.description for c in out["checklist"]]
        assert descriptions.count("Verify date consistency across all documents") == 1
        assert len(descriptions) == 2  # prior + 1 fresh fallback item
        assert out["replan_count"] == 1
