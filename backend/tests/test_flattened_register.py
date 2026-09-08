"""Rebuilding a bank register whose table was flattened one cell per line.

Fixture is the real page-1 text of the Q1 statement that produced the bug: no ruling
lines (so find_tables returns nothing) and description/amount merged into one span
(so column clustering cannot separate them). Reading the flattened stream linearly,
extraction paired each amount with the FOLLOWING row's ref — 380,000 became TXN-010's,
75,000 became TXN-004's — and summed 517,930 against a stated Total Debits of 1,185,700.
"""
import pytest

from app.services.document_processor import _join_cells, _reassemble_flattened_rows

PAGE = """\
Account Holder: Delta Trading Group S.A.E.
Statement Period: 01 Jan – 31 Mar 2025
Opening Balance (01 Jan)
EGP 320,000.00
Total Credits
EGP 1,240,000.00
Total Debits
EGP 1,185,700.00
Closing Balance (31 Mar)
EGP 374,300.00
Date
Ref
No.
Description / الوصفDebit
(EGP)
Credit
(EGP)
Balance
(EGP)
Note
01/01/2025
—
Opening Balance / رصيد افتتاحي—
—
320,000.00
03/01/2025
TXN-
001
Sales Revenue – Invoice SI-100 (مبيعات)
480,000.00
800,000.00
05/01/2025
TXN-
002
Payment to Nile Financial Consulting – Advance
دفعة مقدمة للنيل للاستشارات - عقد CTR-2024-044
50,000.00
750,000.00
⚑ CTR-2024-
044
12/01/2025
TXN-
003
Office Rent Q1 2025 / إيجار المكتب75,000.00
675,000.00
20/01/2025
TXN-
004
Supplier Payment – Al Wadi Imports (مورد)
120,000.00
555,000.00
31/01/2025
TXN-
005
Payment to Nile Financial – Invoice INV-2025-
001 M1
سداد فاتورة الاستشارات - المرحلة الأولى
68,400.00
486,600.00
⚑ INV-2025-
001
05/02/2025
TXN-
006
Sales Revenue – Invoice SI-101 (مبيعات)
380,000.00
866,600.00
14/02/2025
TXN-
007
Staff Salaries February 2025 / رواتب فبراير210,000.00
656,600.00
28/02/2025
TXN-
008
Payment to Nile Financial – Invoice INV-2025-
001 M2
62,930.00
593,670.00
10/03/2025
TXN-
009
Sales Revenue – Invoice SI-102 (مبيعات)
380,000.00
973,670.00
15/03/2025
TXN-
010
Utility Bills Q1 / فواتير مرافق18,000.00
955,670.00
20/03/2025
TXN-
011
Staff Salaries March 2025 / رواتب مارس210,000.00
745,670.00
25/03/2025
TXN-
012
Payment to Nile Consulting – Final / الدفعة الأخيرة
45,370.00
700,300.00
31/03/2025
TXN-
013
Tax Payment Q1 2025 / ضريبة الربع الأول326,000.00
374,300.00
31/03/2025
—
Closing Balance / الرصيد الختامي—
—
374,300.00
"""

# Every debit row, and the figure that row actually carries.
DEBITS = {
    "TXN-002": 50_000.00, "TXN-003": 75_000.00, "TXN-004": 120_000.00,
    "TXN-005": 68_400.00, "TXN-007": 210_000.00, "TXN-008": 62_930.00,
    "TXN-010": 18_000.00, "TXN-011": 210_000.00, "TXN-012": 45_370.00,
    "TXN-013": 326_000.00,
}
STATED_TOTAL_DEBITS = 1_185_700.00


@pytest.fixture(scope="module")
def rebuilt():
    result = _reassemble_flattened_rows(PAGE)
    assert result is not None, "register was not recognised"
    return result


def _row_for(rows: list[str], ref: str) -> str:
    matches = [r for r in rows if ref in r]
    assert len(matches) == 1, f"{ref} appeared in {len(matches)} rows: {matches}"
    return matches[0]


class TestRegisterReassembly:
    def test_header_prose_is_kept_out_of_the_rows(self, rebuilt):
        prose, rows = rebuilt
        assert "Total Debits" in prose
        assert not any("Account Holder" in r for r in rows)

    def test_one_row_per_transaction(self, rebuilt):
        _, rows = rebuilt
        assert len(rows) == 15  # 13 transactions + opening and closing balance

    def test_split_refs_are_healed(self, rebuilt):
        """"TXN-" and "003" arrive on separate lines and must rejoin."""
        _, rows = rebuilt
        assert "TXN-003" in _row_for(rows, "TXN-003")
        assert "CTR-2024-044" in _row_for(rows, "CTR-2024-044")

    def test_amounts_do_not_bleed_between_rows(self, rebuilt):
        """Each ref kept its own figure — the exact mis-pairings that occurred."""
        _, rows = rebuilt
        for ref, wrong in (("TXN-010", "380,000.00"),   # stolen from TXN-009's credit
                           ("TXN-004", "75,000.00"),    # stolen from TXN-003
                           ("TXN-009", "18,000.00")):   # TXN-010's debit
            assert wrong not in _row_for(rows, ref)

    def test_debits_reconcile_with_the_statements_own_total(self, rebuilt):
        """The whole point: summing one debit per row must equal the stated total."""
        _, rows = rebuilt
        total = 0.0
        for ref, amount in DEBITS.items():
            assert f"{amount:,.2f}" in _row_for(rows, ref)
            total += amount
        assert total == STATED_TOTAL_DEBITS

    def test_credit_rows_are_present_and_separate(self, rebuilt):
        _, rows = rebuilt
        for ref, amount in (("TXN-001", "480,000.00"), ("TXN-006", "380,000.00"),
                            ("TXN-009", "380,000.00")):
            assert amount in _row_for(rows, ref)


class TestGuards:
    def test_prose_without_register_words_is_left_alone(self):
        text = "\n".join(["01/01/2025", "Clause one", "02/01/2025", "Clause two",
                          "03/01/2025", "Clause three"])
        assert _reassemble_flattened_rows(text) is None

    def test_too_few_dated_rows_is_left_alone(self):
        text = "Debit Credit Balance\n01/01/2025\n100,000.00\n02/01/2025\n200,000.00"
        assert _reassemble_flattened_rows(text) is None

    def test_rows_without_amounts_are_left_alone(self):
        text = "\n".join(["Debit Credit Balance", "01/01/2025", "a", "02/01/2025", "b",
                          "03/01/2025", "c", "04/01/2025", "d"])
        assert _reassemble_flattened_rows(text) is None


class TestJoinCells:
    def test_hyphen_split_identifier_is_healed(self):
        assert _join_cells(["TXN-", "003", "Rent"]) == "TXN-003 | Rent"

    def test_ordinary_cells_are_pipe_joined(self):
        assert _join_cells(["01/01/2025", "Rent", "75,000.00"]) == "01/01/2025 | Rent | 75,000.00"

    def test_trailing_hyphen_with_non_alnum_next_is_not_merged(self):
        assert _join_cells(["Final –", "45,370.00"]) == "Final – | 45,370.00"


# ── Generalisation beyond the one statement that produced the bug ─────────────

def _register(rows: list[str]) -> str:
    return "Date Ref Description Debit Credit Balance\n" + "\n".join(rows)


class TestDateFormats:
    """Registers in the wild are not all DD/MM/YYYY."""

    FORMATS = [
        ["01/01/2025", "02/01/2025", "03/01/2025"],          # slash
        ["2025-01-01", "2025-01-02", "2025-01-03"],          # ISO
        ["01-01-2025", "02-01-2025", "03-01-2025"],          # dash
        ["01.01.2025", "02.01.2025", "03.01.2025"],          # dotted, full year
        ["15 Jan 2025", "16 Jan 2025", "17 Jan 2025"],       # text month
        ["Jan 15, 2025", "Jan 16, 2025", "Jan 17, 2025"],    # month first
        ["١٥/٠٣/٢٠٢٥", "١٦/٠٣/٢٠٢٥", "١٧/٠٣/٢٠٢٥"],          # Arabic-Indic digits
        ["15 مارس 2025", "16 مارس 2025", "17 مارس 2025"],     # Arabic month
    ]

    def test_row_starts_are_recognised(self):
        for dates in self.FORMATS:
            text = _register([f"{d}\nTXN-\n00{i}\nPayment\n{i}0,000.00\n900,000.00"
                              for i, d in enumerate(dates, start=1)])
            result = _reassemble_flattened_rows(text)
            assert result is not None, f"not recognised: {dates[0]}"
            _, rows = result
            assert len(rows) == 3, dates[0]
            for i in range(1, 4):
                assert f"TXN-00{i}" in rows[i - 1] and f"{i}0,000.00" in rows[i - 1]

    def test_numbered_clauses_are_not_read_as_dates(self):
        """"1.5.2" is a contract clause, not 1 May 2002."""
        text = _register(["1.5.2\nScope of work\n10,000.00",
                          "1.5.3\nDeliverables\n20,000.00",
                          "1.5.4\nAcceptance\n30,000.00"])
        assert _reassemble_flattened_rows(text) is None


class TestDualDateRegisters:
    def test_posting_and_value_date_stay_on_one_row(self):
        """Two date columns open two segments per row; they must rejoin."""
        text = _register([
            "01/01/2025\n03/01/2025\nTXN-\n001\nOpening transfer\n50,000.00\n750,000.00",
            "05/01/2025\n07/01/2025\nTXN-\n002\nOffice rent\n75,000.00\n675,000.00",
            "09/01/2025\n11/01/2025\nTXN-\n003\nSupplier payment\n120,000.00\n555,000.00",
        ])
        result = _reassemble_flattened_rows(text)
        assert result is not None
        _, rows = result
        assert len(rows) == 3
        for ref, amount in (("TXN-001", "50,000.00"), ("TXN-002", "75,000.00"),
                            ("TXN-003", "120,000.00")):
            row = _row_for(rows, ref)
            assert amount in row
            assert row.count("|") >= 5  # both dates plus the row's own cells
