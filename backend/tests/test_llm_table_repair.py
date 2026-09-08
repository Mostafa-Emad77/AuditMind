"""LLM table repair is accepted only when it conserves every number.

Free-form LLM row pairing is what produced the wrong bank total, so the model is
allowed to re-partition a flattened table only under a checksum: rebuilding rows is
a pure re-partition, so the output must contain exactly the input's numbers.
"""

import app.services.document_processor as dp

FLAT = """\
Nile Trading Balance Sheet
As at 31 December 2025
Cash and equivalents
1,200,000.00
Accounts receivable
850,000.00
Inventory
430,000.00
Total assets
2,480,000.00
Accounts payable
610,000.00
Long-term debt
900,000.00
Total liabilities
1,510,000.00
Share capital
970,000.00
Total equity
970,000.00
"""

GOOD = """\
Nile Trading Balance Sheet
As at 31 December 2025
Cash and equivalents | 1,200,000.00
Accounts receivable | 850,000.00
Inventory | 430,000.00
Total assets | 2,480,000.00
Accounts payable | 610,000.00
Long-term debt | 900,000.00
Total liabilities | 1,510,000.00
Share capital | 970,000.00
Total equity | 970,000.00
"""


class _Reply:
    def __init__(self, content): self.content = content


def _stub_llm(monkeypatch, content):
    monkeypatch.setattr(dp, "get_llm", lambda **kw: type(
        "L", (), {"invoke": staticmethod(lambda _p: _Reply(content))})())


class TestTriggerHeuristic:
    def test_flattened_table_is_recognised(self):
        assert dp._looks_flattened_table(FLAT)

    def test_prose_is_not(self):
        prose = "\n".join(
            ["This Agreement is entered into between the parties named below and "
             "sets out the scope of the engagement."] * 6
        )
        assert not dp._looks_flattened_table(prose)

    def test_short_page_is_not(self):
        assert not dp._looks_flattened_table("Total\n100,000.00\nBalance\n50,000.00")


class TestNumberConservation:
    def test_identical_numbers_regardless_of_formatting(self):
        assert dp._number_multiset("1,200,000.00 and 850000") == \
               dp._number_multiset("1200000 and 850,000.00")

    def test_counts_duplicates(self):
        assert dp._number_multiset("5,000.00 5,000.00") != dp._number_multiset("5,000.00")


class TestRepairAcceptance:
    def test_faithful_reconstruction_is_accepted(self, monkeypatch):
        _stub_llm(monkeypatch, GOOD)
        result = dp._llm_repair_flattened_table(FLAT)
        assert result is not None
        prose, rows = result
        assert "Nile Trading Balance Sheet" in prose
        assert any("Total assets | 2,480,000.00" in r for r in rows)
        assert len(rows) == 9  # every line carrying a figure

    def test_unfaithful_reconstruction_is_rejected(self, monkeypatch):
        """Any drop, invention, alteration or duplication must fail the checksum."""
        for bad, reason in (
            (GOOD.replace("Inventory | 430,000.00\n", ""), "a row was dropped"),
            (GOOD + "Goodwill | 300,000.00\n", "a figure was invented"),
            (GOOD.replace("2,480,000.00", "2,490,000.00"), "a figure was altered"),
            (GOOD + "Total assets | 2,480,000.00\n", "a row was duplicated"),
        ):
            _stub_llm(monkeypatch, bad)
            assert dp._llm_repair_flattened_table(FLAT) is None, reason

    def test_reformatting_alone_is_tolerated(self, monkeypatch):
        """Same values, different rendering — a re-partition, so still valid."""
        _stub_llm(monkeypatch, GOOD.replace("1,200,000.00", "1200000.00"))
        assert dp._llm_repair_flattened_table(FLAT) is not None

    def test_llm_failure_falls_back_quietly(self, monkeypatch):
        def _boom(**kw):
            raise RuntimeError("provider down")
        monkeypatch.setattr(dp, "get_llm", _boom)
        assert dp._llm_repair_flattened_table(FLAT) is None

    def test_disabled_by_setting(self, monkeypatch):
        from app.config import get_settings
        settings = get_settings()
        monkeypatch.setattr(settings, "extraction_llm_table_repair", False)
        monkeypatch.setattr(dp, "get_settings", lambda: settings)
        assert dp._llm_repair_flattened_table(FLAT) is None
