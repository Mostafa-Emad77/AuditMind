"""Self-reference validation, graph-pair dedupe, and always-on semantic retrieval."""
import json

from app.agents.cross_checker.agent import _dedupe_graph_pairs
from app.services import graph_builder as gb
from app.tools import hybrid_retriever as hr
from app.tools.planner_tools import generate_checklist


# ── graph pair dedupe before top-N ───────────────────────────────────────────

def _pair(v1, v2, reason, d1="a", d2="b"):
    return {"amount_value1": v1, "amount_value2": v2, "comparison_reason": reason,
            "doc1_id": d1, "doc2_id": d2}


class TestGraphPairDedupe:
    def test_same_numeric_pair_collapses_across_doc_combinations(self):
        rows = [
            _pair(68400.0, 25000.0, "retainer rate vs bank payment", "bank", "contract"),
            _pair(68400.0, 25000.0, "retainer rate vs bank payment", "bank", "inv2"),
            _pair(25000.0, 68400.0, "retainer rate vs bank payment", "contract", "bank"),  # mirrored
            _pair(180000.0, 200000.0, "contract total vs invoice's stated contract reference"),
        ]
        out = _dedupe_graph_pairs(rows)
        assert len(out) == 2
        assert {round(out[1]["amount_value1"]), round(out[1]["amount_value2"])} == {180000, 200000}

    def test_same_numbers_different_reason_are_kept(self):
        rows = [_pair(1.0, 2.0, "x"), _pair(1.0, 2.0, "y")]
        assert len(_dedupe_graph_pairs(rows)) == 2


# ── find_reference_mismatches ────────────────────────────────────────────────

def _row(claim_value, truth_value, truth_role="total_contract_value", truth_doc_type="contract",
         claim_doc="inv1", truth_doc="ctr", claim_cur="EGP", truth_cur="EGP"):
    return {
        "claim_role": "invoice_referenced_contract_value", "claim_value": claim_value,
        "claim_raw": f"EGP {claim_value:,.0f}", "claim_currency": claim_cur, "claim_page": 1,
        "claim_doc_id": claim_doc, "claim_doc_name": f"{claim_doc}.pdf", "claim_doc_type": "invoice",
        "anchor_value": "CTR-2024-044", "anchor_type": "contract_id",
        "truth_role": truth_role, "truth_value": truth_value, "truth_raw": f"EGP {truth_value:,.0f}",
        "truth_currency": truth_cur, "truth_page": 2, "truth_doc_id": truth_doc,
        "truth_doc_name": f"{truth_doc}.pdf", "truth_doc_type": truth_doc_type,
    }


class TestFindReferenceMismatches:
    def _run(self, monkeypatch, rows):
        captured = {}

        class _S:
            def __enter__(self): return self
            def __exit__(self, *e): return False
            def run(self, query, **kw):
                captured["query"] = query; captured["params"] = kw
                return list(rows)

        class _D:
            def session(self, **kw): return _S()

        monkeypatch.setattr(gb, "_run_with_reconnect", lambda fn: fn(_D()))
        out = gb.find_reference_mismatches(["inv1", "ctr"])
        return out, captured

    def test_restated_contract_value_mismatch_is_reported(self, monkeypatch):
        out, cap = self._run(monkeypatch, [_row(180000.0, 200000.0)])
        assert len(out) == 1
        assert out[0]["difference"] == -20000.0
        assert abs(out[0]["relative_difference"] - 0.10) < 1e-9
        assert out[0]["label"] == "contract value"
        assert "invoice_referenced_contract_value" in cap["params"]["claim_roles"]

    def test_matching_restatement_is_silent(self, monkeypatch):
        out, _ = self._run(monkeypatch, [_row(200000.0, 200000.0)])
        assert out == []

    def test_only_the_referenced_documents_own_figure_counts_as_truth(self, monkeypatch):
        """A bank statement echoing the contract value is not the contract."""
        out, _ = self._run(monkeypatch, [
            _row(180000.0, 200000.0, truth_doc_type="bank_statement", truth_doc="bank"),
            _row(180000.0, 150000.0, truth_role="invoice_referenced_contract_value",
                 truth_doc_type="invoice", truth_doc="inv2"),
        ])
        assert out == []

    def test_currency_mismatch_is_not_compared(self, monkeypatch):
        out, _ = self._run(monkeypatch, [_row(180000.0, 200000.0, claim_cur="USD")])
        assert out == []

    def test_duplicate_rows_collapse(self, monkeypatch):
        out, _ = self._run(monkeypatch, [_row(180000.0, 200000.0)] * 3)
        assert len(out) == 1

    def test_empty_doc_ids(self):
        assert gb.find_reference_mismatches([]) == []


# ── planner always includes the self-reference check ─────────────────────────

class TestSelfReferenceChecklistItem:
    @staticmethod
    def _no_llm(monkeypatch):
        """Force the deterministic checklist path so the test never hits a live model."""
        import app.tools.planner_tools as pt
        monkeypatch.setattr(pt, "_generate_checklist_llm", lambda *a, **k: [])

    def test_present_for_multi_document_audits(self, monkeypatch):
        self._no_llm(monkeypatch)
        raw = generate_checklist.invoke({
            "doc_types_json": json.dumps(["invoice", "contract"]),
            "doc_ids_json": json.dumps(["a", "b"]),
        })
        items = json.loads(raw)["checklist"]
        assert items[0]["check_type"] == "self_reference"
        assert items[0]["priority"] == "high"
        assert sum(1 for c in items if c["check_type"] == "self_reference") == 1

    def test_absent_for_single_document(self, monkeypatch):
        self._no_llm(monkeypatch)
        raw = generate_checklist.invoke({
            "doc_types_json": json.dumps(["invoice"]),
            "doc_ids_json": json.dumps(["a"]),
        })
        assert not any(c["check_type"] == "self_reference" for c in json.loads(raw)["checklist"])


# ── hybrid retriever: semantic search always runs ────────────────────────────

class TestRetrievalAlwaysSemantic:
    def _wire(self, monkeypatch, graph_hits):
        calls = {"semantic": 0}

        def _sem(query, doc_ids=None, top_k=10):
            calls["semantic"] += 1
            return [{"doc_id": "a", "text": "Contract value EGP 200,000", "page_num": 1,
                     "language": "english", "score": 0.4, "chunk_id": "c1"}]

        monkeypatch.setattr(hr, "semantic_search", _sem)
        monkeypatch.setattr(hr, "keyword_search", lambda **kw: [])
        monkeypatch.setattr(hr, "query_graph_for_entities", lambda ents, ids, depth=2: graph_hits)
        return calls

    def test_graph_route_still_returns_semantic_passages(self, monkeypatch):
        calls = self._wire(monkeypatch, graph_hits=[])
        raw = hr.search_hybrid_rag.invoke({
            "query": "Compare the invoices to detect contradictory line items across documents",
            "doc_ids": "a,b", "top_k": 5,
        })
        data = json.loads(raw)
        assert calls["semantic"] == 1
        assert data["vector_count"] == 1
        assert data["total"] >= 1

    def test_graph_hits_are_additive(self, monkeypatch):
        self._wire(monkeypatch, graph_hits=[{"value": "CTR-2024-044", "doc_id": "b", "page": 1,
                                             "entity_type": "contract_id"}])
        raw = hr.search_hybrid_rag.invoke({
            "query": "Reconcile CTR-2024-044 value across documents", "doc_ids": "a,b", "top_k": 5,
        })
        data = json.loads(raw)
        assert data["vector_count"] == 1 and data["graph_count"] == 1
        assert data["route"] == "graph"
