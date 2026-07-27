"""Regression tests for Phase 3 performance changes (3.3, 3.4, 3.5) and the
Phase-1 graph-adjudication blocker that preceded them."""
import app.services.document_processor as dp
import app.services.graph_builder as gb
import app.services.vector_store as vs
from app.agents.cross_checker import agent as cc_agent


# ── Phase-1 blocker: agent.py used parse_monetary_amount without importing it ──
# The NameError was swallowed by the phase's broad `except Exception`, silently
# disabling every graph-sourced finding. Guard the binding directly.

class TestCrossCheckerAgentNames:
    def test_parse_monetary_amount_is_bound_in_agent_module(self):
        assert callable(getattr(cc_agent, "parse_monetary_amount", None))

    def test_it_is_the_real_parser(self):
        assert cc_agent.parse_monetary_amount("50,000 EGP") == 50000.0
        assert cc_agent.parse_monetary_amount("not an amount") is None


# ── 3.3 — role-matrix post-filter on find_contradictions ─────────────────────

def _pair(role1: str, role2: str, **extra) -> dict:
    row = {"role1": role1, "role2": role2, "entity1_id": "a", "entity2_id": "b"}
    row.update(extra)
    return row


class TestFindContradictionsRoleFilter:
    def test_incomparable_role_pairs_are_dropped(self, monkeypatch):
        rows = [
            _pair("total_contract_value", "total_invoice"),   # comparable
            _pair("retainer", "total_contract_value"),        # not comparable
            _pair("single_payment", "total_invoice"),         # not comparable
        ]
        monkeypatch.setattr(gb, "_run_with_reconnect", lambda _fn: rows)

        kept = gb.find_contradictions(["d1", "d2"])

        assert len(kept) == 1
        assert kept[0]["role1"] == "total_contract_value"
        assert kept[0]["role2"] == "total_invoice"

    def test_unknown_roles_are_kept(self, monkeypatch):
        rows = [_pair("unknown", "total_invoice"), _pair("unknown", "unknown")]
        monkeypatch.setattr(gb, "_run_with_reconnect", lambda _fn: rows)

        assert len(gb.find_contradictions(["d1", "d2"])) == 2

    def test_empty_result_is_passed_through(self, monkeypatch):
        monkeypatch.setattr(gb, "_run_with_reconnect", lambda _fn: [])
        assert gb.find_contradictions(["d1", "d2"]) == []


class TestFindContradictionsQuery:
    """The Cypher itself: symmetric-pair dedup + materiality ordering."""

    def _captured_cypher(self, monkeypatch) -> str:
        captured = {}

        class _FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def run(self, query, **kwargs):
                captured["query"] = query
                return []

        class _FakeDriver:
            def session(self, **kwargs):
                return _FakeSession()

        monkeypatch.setattr(gb, "_run_with_reconnect", lambda fn: fn(_FakeDriver()))
        gb.find_contradictions(["d1", "d2"])
        return captured["query"]

    def test_symmetric_pairs_are_deduped(self, monkeypatch):
        query = self._captured_cypher(monkeypatch)
        # Ordering constraint keeps one direction of each unordered pair.
        assert "e1.entity_id < e2.entity_id" in query
        assert "e1.entity_id <> e2.entity_id" not in query

    def test_results_are_ordered_by_materiality(self, monkeypatch):
        query = self._captured_cypher(monkeypatch)
        assert "ORDER BY" in query
        # ORDER BY must precede LIMIT, or the top-50 is still arbitrary.
        assert query.index("ORDER BY") < query.index("LIMIT")


# ── 3.4 — Qdrant client singleton + one-shot collection init ─────────────────

class TestQdrantSingleton:
    def test_repeated_calls_return_the_same_client(self, monkeypatch):
        monkeypatch.setattr(vs, "_client", None)
        created = []

        class _FakeClient:
            def __init__(self, **kwargs):
                created.append(kwargs)

        monkeypatch.setattr(vs, "QdrantClient", _FakeClient)

        first = vs.get_qdrant_client()
        second = vs.get_qdrant_client()

        assert first is second
        assert len(created) == 1


class TestEnsureCollectionRunsOnce:
    def test_second_call_is_a_noop(self, monkeypatch):
        monkeypatch.setattr(vs, "_collection_ready", None)
        calls = []

        class _FakeClient:
            def get_collections(self):
                calls.append("get_collections")
                return type("R", (), {"collections": [type("C", (), {"name": "docs"})()]})()

            def create_payload_index(self, **kwargs):
                calls.append("create_payload_index")

        client = _FakeClient()
        vs.ensure_collection(client, "docs", 768)
        first_round = len(calls)
        assert first_round > 0

        vs.ensure_collection(client, "docs", 768)
        assert len(calls) == first_round, "repeat call should short-circuit"


# ── 3.5 — heuristic-first document classification ────────────────────────────

class TestClassificationOrder:
    def test_confident_heuristic_skips_the_llm(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            dp, "_classify_document_type_llm",
            lambda text, filename: called.append(1) or "invoice",
        )

        text = "Bank Statement — IBAN EG12345 — opening balance / closing balance, transactions, debit credit"
        assert dp._classify_document_type(text, "stmt.pdf") == "bank_statement"
        assert not called, "LLM must not be called when the heuristic is confident"

    def test_unknown_heuristic_falls_back_to_llm(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            dp, "_classify_document_type_llm",
            lambda text, filename: (called.append(1), "contract")[1],
        )

        assert dp._classify_document_type("qqq zzz", "x.pdf") == "contract"
        assert called, "LLM must be called when the heuristic returns unknown"

    def test_both_uncertain_yields_unknown(self, monkeypatch):
        monkeypatch.setattr(dp, "_classify_document_type_llm", lambda text, filename: None)
        assert dp._classify_document_type("qqq zzz", "x.pdf") == "unknown"
