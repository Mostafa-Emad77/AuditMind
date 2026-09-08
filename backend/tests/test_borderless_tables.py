"""Borderless tables must still be lifted as whole rows.

A statement drawn without ruling lines returned nothing from find_tables' default
"lines" strategy, so its register fell through to flattened one-cell-per-line prose
and the entity LLM paired each ref with the neighbouring row's amounts.
"""
import fitz

from app.services.document_processor import _extract_page_tables, _looks_tabular, _page_tables

HEADER = ["Date", "Ref No.", "Description", "Debit (EGP)", "Credit (EGP)", "Balance (EGP)"]
ROWS = [
    ["05/01/2025", "TXN-002", "Payment to Nile Advance", "50,000.00", "", "750,000.00"],
    ["12/01/2025", "TXN-003", "Office Rent Q1 2025", "75,000.00", "", "675,000.00"],
    ["20/01/2025", "TXN-004", "Supplier Payment Al Wadi", "120,000.00", "", "555,000.00"],
    ["31/01/2025", "TXN-005", "Payment to Nile INV M1", "68,400.00", "", "486,600.00"],
]
_COLS = [40, 110, 190, 380, 460, 540]


def _page(build) -> fitz.Page:
    doc = fitz.open()
    build(doc.new_page())
    data = doc.tobytes()
    doc.close()
    return fitz.open(stream=data, filetype="pdf")[0]


def _borderless_register(page: fitz.Page) -> None:
    y = 80
    for cell, x in zip(HEADER, _COLS):
        page.insert_text((x, y), cell, fontsize=8)
    for row in ROWS:
        y += 22
        for cell, x in zip(row, _COLS):
            page.insert_text((x, y), cell, fontsize=8)


def _prose(page: fitz.Page) -> None:
    page.insert_textbox(
        fitz.Rect(50, 50, 550, 320),
        "This Agreement is made between Delta Trading Group S.A.E. and Nile Financial "
        "Consulting. The parties agree to the terms and conditions set out herein, "
        "including scope, deliverables, payment and termination provisions.",
        fontsize=10,
    )


class TestBorderlessTables:
    def test_default_lines_strategy_finds_nothing(self):
        """Precondition: this is exactly the case the original implementation missed."""
        page = _page(_borderless_register)
        assert len(page.find_tables(strategy="lines").tables) == 0

    def test_text_strategy_recovers_the_register(self):
        page = _page(_borderless_register)
        assert len(_page_tables(page)) >= 1

    def test_each_ref_stays_on_one_line_with_its_own_amount(self):
        """The off-by-one bug: a ref must never be separated from its own figures."""
        _, blocks = _extract_page_tables(_page(_borderless_register))
        lines = [ln for _, rows in blocks for ln in rows]
        for ref, amount in (("TXN-002", "50,000.00"), ("TXN-003", "75,000.00"),
                            ("TXN-004", "120,000.00"), ("TXN-005", "68,400.00")):
            row = next((ln for ln in lines if ref in ln), None)
            assert row is not None, f"{ref} missing from {lines}"
            assert amount in row, ref

    def test_prose_is_not_mistaken_for_a_table(self):
        """Text clustering must not turn paragraphs into rows."""
        assert _page_tables(_page(_prose)) == []


class TestLooksTabular:
    def test_header_row_qualifies(self):
        assert _looks_tabular([HEADER, ["", "", "", "", "", ""]])

    def test_numeric_rows_qualify_without_a_header(self):
        assert _looks_tabular([r[:] for r in ROWS])

    def test_prose_rows_do_not_qualify(self):
        assert not _looks_tabular([
            ["The parties agree", "to the terms"],
            ["including scope", "and deliverables"],
        ])

    def test_single_row_does_not_qualify(self):
        assert not _looks_tabular([ROWS[0]])
