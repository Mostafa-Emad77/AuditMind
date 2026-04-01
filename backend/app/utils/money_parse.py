"""Deterministic monetary amount parsing — single-token extraction, no digit concatenation."""
from __future__ import annotations

import re
from typing import Optional

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Currency / monetary unit tokens (Latin + common Arabic abbreviations)
_CURRENCY = (
    r"USD|US\$|EUR|GBP|EGP|LE|SAR|AED|KWD|OMR|QAR|BHD|"
    r"دولار|يورو|جنيه|ج\.م|ريال|درهم"
)

# Prefer comma-grouped thousands OR plain digit runs (do not truncate 5000 → 500)
_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

# Single amount: optional currency, number with grouping, optional currency
_SINGLE_AMOUNT_RE = re.compile(
    rf"^\s*(?:({_CURRENCY})\s*)?({_NUM})\s*(?:({_CURRENCY}))?\s*$",
    re.IGNORECASE | re.UNICODE,
)

# Number with adjacent currency (strong monetary context)
_WITH_CURRENCY_LEFT = re.compile(
    rf"(?:^|\s)({_CURRENCY})\s*({_NUM})",
    re.IGNORECASE | re.UNICODE,
)
_WITH_CURRENCY_RIGHT = re.compile(
    rf"({_NUM})\s*({_CURRENCY})(?:\s|$)",
    re.IGNORECASE | re.UNICODE,
)


def _comma_float(s: str) -> Optional[float]:
    s = s.strip().replace(",", "")
    if not s or s == ".":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _is_probable_year_token(s: str, val: float) -> bool:
    if "." in s:
        return False
    digits = re.sub(r"\D", "", s)
    if len(digits) != 4:
        return False
    return 1900 <= val <= 2100


def parse_monetary_amount(value: str) -> Optional[float]:
    """
    Parse exactly one monetary value from a string.
    Does not concatenate disjoint digit runs (avoids synthetic huge numbers from IDs/filenames).
    """
    if not value or not str(value).strip():
        return None
    s = str(value).translate(_ARABIC_INDIC).strip()
    if not s:
        return None

    m = _SINGLE_AMOUNT_RE.match(s)
    if m:
        return _comma_float(m.group(2))

    candidates: list[float] = []
    for mm in _WITH_CURRENCY_LEFT.finditer(s):
        num_part = mm.group(2)
        v = _comma_float(num_part)
        if v is not None and not _is_probable_year_token(num_part, v):
            candidates.append(v)
    for mm in _WITH_CURRENCY_RIGHT.finditer(s):
        num_part = mm.group(1)
        v = _comma_float(num_part)
        if v is not None and not _is_probable_year_token(num_part, v):
            candidates.append(v)

    if candidates:
        uniq = sorted(set(round(c, 8) for c in candidates))
        if len(uniq) == 1:
            return uniq[0]
        return None

    # Isolated digit runs (no currency): only accept if exactly one non-year run
    runs = re.findall(r"\d[\d,]*\.?\d*", s)
    parsed_runs: list[tuple[str, float]] = []
    for r in runs:
        v = _comma_float(r)
        if v is None:
            continue
        if _is_probable_year_token(r, v):
            continue
        parsed_runs.append((r, v))

    if len(parsed_runs) == 1:
        return parsed_runs[0][1]
    if len(parsed_runs) == 0:
        return None
    values = {round(t[1], 8) for t in parsed_runs}
    if len(values) == 1:
        return next(iter(values))
    return None


def amounts_within_tiny_tolerance(a: float, b: float) -> bool:
    """True if difference is negligible (rounding / float noise)."""
    diff = abs(a - b)
    if diff == 0:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return diff <= max(1e-6 * scale, 0.01)
