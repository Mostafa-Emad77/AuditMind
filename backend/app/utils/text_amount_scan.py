"""Collect monetary-looking digit groups from text without regex (character scan)."""
from __future__ import annotations

from app.utils.money_parse import parse_monetary_amount


def iter_digit_group_strings(text: str) -> list[str]:
    """
    Yield maximal contiguous runs of digits with optional comma thousands and one decimal part.
    Example: "paid 226,700.00 vs 200000" -> ["226,700.00", "200000"].
    """
    if not text:
        return []
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch.isdigit() or ch == "," or ch == ".":
            buf.append(ch)
        else:
            if buf:
                s = "".join(buf)
                if any(c.isdigit() for c in s):
                    out.append(s)
                buf = []
    if buf:
        s = "".join(buf)
        if any(c.isdigit() for c in s):
            out.append(s)
    return out


def _is_probable_year_token(raw: str, value: float) -> bool:
    if "." in raw:
        return False
    digits_only = "".join(c for c in raw if c.isdigit())
    if len(digits_only) != 4:
        return False
    try:
        iv = int(digits_only)
    except ValueError:
        return False
    return 1900 <= iv <= 2035 and abs(value - iv) < 0.01


def amounts_from_text_scan(text: str, *, min_value: float = 1000.0) -> list[float]:
    """
    Parse amounts by scanning digit groups and delegating to parse_monetary_amount.
    Skips bare 4-digit calendar years.
    """
    nums: list[float] = []
    for raw in iter_digit_group_strings(text):
        p = parse_monetary_amount(raw)
        if p is None or p < min_value:
            continue
        if _is_probable_year_token(raw, p):
            continue
        nums.append(p)
    return nums
