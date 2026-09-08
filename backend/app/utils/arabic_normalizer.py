"""Arabic text normalization utilities."""
import re
from typing import Any

from app.utils.money_parse import parse_monetary_amount

# Unicode ranges for Arabic script
_ARABIC_RANGE = range(0x0600, 0x06FF + 1)
_ARABIC_SUPPLEMENT = range(0x0750, 0x077F + 1)

# Tashkeel (diacritics) codepoints
_TASHKEEL = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)

# Alef variants → bare alef
_ALEF_VARIANTS = re.compile("[إأآا]")

# Yaa variants → dotless yaa
_YAA_VARIANTS = re.compile("[يى]")

# Taa marbuta → haa
_TAA_MARBUTA = re.compile("ة")

# Kashida (tatweel)
_KASHIDA = re.compile("\u0640")

# Monetary context (Arabic + Latin) — amount must appear near currency OR these keywords
_MONETARY_KEYWORDS = re.compile(
    r"(?:المبلغ|إجمالي|القيمة|المستحق|المطلوب|ثمن|سعر|تكلفة|دفعة|مدفوع|مبلغ|قيمة\s*العقد|"
    r"amount|total|sum|payment|payable|invoice|contract\s+value|balance|due|EGP|USD|\$)",
    re.IGNORECASE | re.UNICODE,
)

_CURRENCY = (
    r"جنيه|ج\.م|EGP|USD|US\$|EUR|GBP|SAR|AED|ريال|دولار|يورو|درهم|LE\b"
)

# Strong patterns: digits + currency or currency + digits (both directions)
_NUM_AR = r"[\d٠-٩]+(?:[،,]\d{3})*(?:\.\d+)?"
_PATTERN_AR_AMOUNT_FIRST = re.compile(
    rf"({_NUM_AR})\s*({_CURRENCY})",
    re.IGNORECASE | re.UNICODE,
)
_PATTERN_AR_CURRENCY_FIRST = re.compile(
    rf"({_CURRENCY})\s*({_NUM_AR})",
    re.IGNORECASE | re.UNICODE,
)
_NUM_EN = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,4})?"
_PATTERN_EN_NUMBER_CURRENCY = re.compile(
    rf"\b({_NUM_EN})\s*({_CURRENCY})\b",
    re.IGNORECASE | re.UNICODE,
)
_PATTERN_EN_CURRENCY_NUMBER = re.compile(
    rf"\b({_CURRENCY})\s*({_NUM_EN})\b",
    re.IGNORECASE | re.UNICODE,
)

# Reference / year patterns to exclude as standalone "amounts"
_REF_YEAR = re.compile(
    r"(?:ref|reference|رقم|ص\s*[/]\s*م|invoice\s*#|contract\s*#|عقد\s*رقم)\s*[:\-]?\s*",
    re.IGNORECASE | re.UNICODE,
)


def is_arabic_text(text: str, threshold: float = 0.3) -> bool:
    """Return True if Arabic characters make up >= threshold of the text."""
    if not text:
        return False
    arabic_chars = sum(1 for c in text if ord(c) in _ARABIC_RANGE or ord(c) in _ARABIC_SUPPLEMENT)
    return arabic_chars / max(len(text), 1) >= threshold


def detect_language(text: str) -> str:
    """Return 'arabic', 'english', or 'mixed' for a text snippet."""
    if not text.strip():
        return "unknown"
    arabic_chars = sum(1 for c in text if ord(c) in _ARABIC_RANGE)
    latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    total = arabic_chars + latin_chars
    if total == 0:
        return "unknown"
    ratio = arabic_chars / total
    if ratio > 0.7:
        return "arabic"
    if ratio < 0.3:
        return "english"
    return "mixed"


def normalize_arabic(text: str) -> str:
    """Strip tashkeel/kashida, unify alef/yaa (and taa marbuta by flag), collapse whitespace."""
    text = _TASHKEEL.sub("", text)
    text = _ALEF_VARIANTS.sub("ا", text)
    text = _YAA_VARIANTS.sub("ي", text)
    text = _KASHIDA.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_mixed_text(text: str) -> str:
    """Normalize a document chunk that may contain both Arabic and English."""
    lines = text.split("\n")
    normalized_lines = []
    for line in lines:
        if is_arabic_text(line, threshold=0.2):
            normalized_lines.append(normalize_arabic(line))
        else:
            # Basic English normalization: collapse whitespace, strip
            normalized_lines.append(re.sub(r"\s+", " ", line).strip())
    return "\n".join(normalized_lines)


def _window_has_monetary_context(text: str, start: int, end: int, span: int = 48) -> bool:
    lo = max(0, start - span)
    hi = min(len(text), end + span)
    window = text[lo:hi]
    return bool(_MONETARY_KEYWORDS.search(window))


def _overlaps_reference_year(text: str, start: int, end: int) -> bool:
    """True if match is immediately after a reference/year label (likely ID, not money)."""
    prefix = text[max(0, start - 32) : start]
    if _REF_YEAR.search(prefix):
        return True
    chunk = text[start:end]
    if re.fullmatch(r"[\d,٬،.\s٠-٩]+", chunk) and len(re.sub(r"\D", "", chunk)) == 4:
        try:
            y = int(re.sub(r"\D", "", chunk.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))))
            if 1900 <= y <= 2100:
                return True
        except ValueError:
            pass
    return False


def extract_amounts(text: str) -> list[dict[str, Any]]:
    """
    Extract monetary amounts with currency or clear monetary context.
    Each item: raw, position, end, value (parsed float or None).
    """
    seen: set[tuple[int, int]] = set()
    results: list[dict[str, Any]] = []

    def _push(raw: str, start: int, end: int) -> None:
        raw = raw.strip()
        if not raw or (start, end) in seen:
            return
        if _overlaps_reference_year(text, start, end):
            return
        # Require currency in match OR monetary keyword window
        has_currency = bool(
            re.search(_CURRENCY, raw, re.IGNORECASE | re.UNICODE)
        )
        if not has_currency and not _window_has_monetary_context(text, start, end):
            return
        seen.add((start, end))
        parsed = parse_monetary_amount(raw)
        results.append({
            "raw": raw,
            "position": start,
            "end": end,
            "value": parsed,
        })

    for rx in (
        _PATTERN_AR_AMOUNT_FIRST,
        _PATTERN_AR_CURRENCY_FIRST,
        _PATTERN_EN_NUMBER_CURRENCY,
        _PATTERN_EN_CURRENCY_NUMBER,
    ):
        for m in rx.finditer(text):
            raw = m.group(0).strip()
            if raw and any(c.isdigit() or c in "٠١٢٣٤٥٦٧٨٩" for c in raw):
                _push(raw, m.start(), m.end())

    results.sort(key=lambda x: x["position"])
    return results
