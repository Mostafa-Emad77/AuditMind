"""Document ingestion pipeline: PDF → OCR → normalize → chunk → Qdrant."""
import io
import logging
import re
import uuid
from typing import Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.schemas import DocumentMeta, DocumentChunk
from app.utils.arabic_normalizer import (
    detect_language,
    normalize_mixed_text,
)
from app.utils.llm_factory import get_llm
from app.utils.llm_json import extract_json_obj as _extract_json_obj, normalize_llm_content
from app.services.vector_store import store_chunks

logger = logging.getLogger(__name__)

# Lazy-loaded OCR reader (EasyOCR)
_easyocr_reader = None

# Minimum characters per page to consider text "good" (not garbage OCR)
_MIN_CHARS_THRESHOLD = 50

# Chunking settings
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 150

_DOC_TYPES = {
    "invoice", "contract", "balance_sheet", "bank_statement", "audit_report",
    "payment_certificate", "qs_report", "board_resolution", "unknown",
}


def _is_scanned_page(page_text: str) -> bool:
    """Heuristic: if fewer than threshold chars, page is likely a scanned image."""
    return len(page_text.strip()) < _MIN_CHARS_THRESHOLD


def _get_easyocr_reader():
    """
    Lazy-initialize EasyOCR reader once per process.
    Uses Arabic + English recognition for mixed MENA documents.
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["ar", "en"], gpu=False)
    return _easyocr_reader


def _ocr_page(page: fitz.Page) -> str:
    """Run EasyOCR (ar+en), then ArabicOCR as fallback."""
    try:
        # Render to image at 300 DPI
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")

        # 1) EasyOCR first (better multilingual OCR for mixed Arabic/English docs)
        try:
            import numpy as np
            from PIL import Image

            reader = _get_easyocr_reader()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img)
            result_lines = reader.readtext(img_np, detail=0, paragraph=True)
            result = "\n".join([line for line in result_lines if isinstance(line, str)])
            if result and len(result.strip()) >= _MIN_CHARS_THRESHOLD:
                return result
        except Exception as e:
            logger.debug("EasyOCR failed, trying ArabicOCR: %s", e)

        # 2) ArabicOCR fallback
        try:
            from arabicOcr import arabicOcr
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
            result = arabicOcr(tmp_path)
            os.unlink(tmp_path)
            if result and len(result.strip()) >= _MIN_CHARS_THRESHOLD:
                return result
        except Exception as e:
            logger.debug("ArabicOCR failed after EasyOCR: %s", e)
        return ""

    except Exception as e:
        logger.error("OCR failed for page: %s", e)
        return ""


# Identity tokens in the filename / title block beat vocabulary scoring (a payment
# certificate is full of invoice words). Most-specific first; first match wins.
_TITLE_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("payment_certificate", r"payment[\s_-]*cert|interim[\s_-]*payment|\bipc\b|\bpc[-\s]?\d{4}[-\s]?\d+"
                            r"|شهاد[ةه]\s*(استحقاق\s*)?(ال)?دفع|مستخلص\s*(دفع|رقم)"),
    ("board_resolution",    r"board[\s_-]*resolution|board\s*(of\s*directors\s*)?(meeting\s*)?minutes"
                            r"|\bbr[-\s]?\d{4}[-\s]?\d+|محضر\s*اجتماع\s*مجلس|قرار\s*مجلس\s*(ال)?[إا]دار[ةه]"),
    ("qs_report",           r"\bqs[\s_-]*(report|rpt)|quantity[\s_-]*survey|certification[\s_-]*report"
                            r"|\bqs[-\s]?rpt[-\s]?\d+|مساح[ةه]\s*(ال)?كميات|تقرير\s*(ال)?كميات"),
    ("bank_statement",      r"bank[\s_-]*statement|account[\s_-]*statement|كشف\s*حساب"),
    # invoice before contract: invoices cite the contract they bill against.
    ("invoice",             r"\binvoice\b|tax\s*invoice|\binv[-\s]?[a-z]*[-\s]?\d{4}|فاتور[ةه]"),
    ("contract",            r"\bcontract\b|\bagreement\b|\bsubcontract\b|عقد\s*(مقاول[ةه]|من\s*الباطن)?"),
)
_TITLE_WINDOW = 600  # chars of the document head treated as its title block


def _classify_by_title(text: str, filename: str) -> str | None:
    """Return a doc_type if the filename or title block names one unambiguously."""
    fname = re.sub(r"[^a-z0-9؀-ۿ]+", " ", (filename or "").lower())
    head = (text or "")[:_TITLE_WINDOW].lower()
    for doc_type, pattern in _TITLE_OVERRIDES:
        if re.search(pattern, fname, flags=re.IGNORECASE):
            return doc_type
    for doc_type, pattern in _TITLE_OVERRIDES:
        if re.search(pattern, head, flags=re.IGNORECASE):
            return doc_type
    return None


def _classify_document_type(text: str, filename: str) -> str:
    """Title block → vocabulary heuristic → LLM (only when neither can decide)."""
    title_type = _classify_by_title(text, filename)
    if title_type:
        return title_type
    heuristic_type = _classify_document_type_heuristic(text, filename)
    if heuristic_type != "unknown":
        return heuristic_type
    return _classify_document_type_llm(text, filename) or "unknown"


def _classify_document_type_llm(text: str, filename: str) -> str | None:
    sample = (text or "").strip()
    if not sample:
        return None
    sample = sample[:5000]
    prompt = (
        "Classify this financial document into exactly one doc_type.\n"
        "Allowed doc_type values:\n"
        "- invoice\n"
        "- contract\n"
        "- balance_sheet\n"
        "- bank_statement\n"
        "- audit_report\n"
        "- payment_certificate  (an interim/payment certificate or valuation certifying an amount payable for work done; NOT an invoice)\n"
        "- qs_report  (a quantity surveyor's measurement / milestone certification report; NOT a contract)\n"
        "- board_resolution  (board of directors meeting minutes or a formal resolution approving something; NOT an invoice)\n"
        "- unknown\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "doc_type": "one_of_allowed_values",\n'
        '  "confidence": 0.0_to_1.0,\n'
        '  "reason": "short reason"\n'
        "}\n\n"
        f"filename: {filename}\n"
        f"text_sample: {sample}"
    )
    try:
        llm = get_llm(temperature=0.0)
        resp = llm.invoke(prompt)
        content = normalize_llm_content(resp)
        data = _extract_json_obj(content)
        doc_type = str((data or {}).get("doc_type", "unknown")).strip().lower()
        if doc_type in _DOC_TYPES:
            return doc_type
    except Exception as e:
        logger.debug("LLM doc_type classification failed for %s: %s", filename, e)
    return None


def _classify_document_type_heuristic(text: str, filename: str) -> str:
    """OCR-tolerant vocabulary classifier over content + filename."""
    text_lower = text.lower()
    # Separators → spaces so \b patterns match "contract_CTR2024044.pdf".
    filename_lower = re.sub(r"[^a-z0-9؀-ۿ]+", " ", filename.lower())

    patterns: dict[str, list[tuple[str, int]]] = {
        "invoice": [
            (r"\binvoice\b", 3),
            (r"\binvoice\s*(no|number|#)\b", 5),
            (r"tax\s*invoice", 4),
            (r"\bsubtotal\b", 3),
            (r"\bvat\b|\bvalue\s*added\s*tax\b", 3),
            (r"\btotal\s*(amount|due|payable)\b", 4),
            (r"\bpo\s*(no|number|#)\b", 2),
            (r"فاتور[هة]|فواتير", 4),
            (r"المبلغ\s*(الاجمالي|الإجمالي|المستحق)", 4),
            (r"ضريب[هة]|ضريبه", 2),
        ],
        "contract": [
            (r"\bcontract\b|\bagreement\b", 4),
            (r"\bbetween\b.*\band\b", 3),
            (r"\beffective\s*date\b|\bterm\b|\btermination\b", 3),
            (r"\bparty\b|\bparties\b", 2),
            (r"عقد|اتفاقي[هة]", 4),
            (r"الطرف\s*(الاول|الأول|الثاني)", 3),
            (r"الشروط\s*والاحكام|الشروط\s*والأحكام", 3),
        ],
        "balance_sheet": [
            (r"balance\s*sheet", 5),
            (r"\bassets?\b", 3),
            (r"\bliabilit(y|ies)\b", 3),
            (r"\bequity\b|\bshareholders?\b", 3),
            (r"الميزاني[هة]", 5),
            (r"الاصول|الأصول", 3),
            (r"الخصوم", 3),
            (r"حقوق\s*الملكي[هة]", 3),
        ],
        "bank_statement": [
            (r"bank\s*statement|account\s*statement", 5),
            (r"\biban\b|\bswift\b|\bbic\b", 3),
            (r"\btransaction(s)?\b", 3),
            (r"\bdebit\b|\bcredit\b", 3),
            (r"\bopening\s*balance\b|\bclosing\s*balance\b", 4),
            (r"كشف\s*حساب", 5),
            (r"مدين|دائن|رصيد", 3),
            (r"تحويل|ايداع|إيداع|سحب", 2),
        ],
        "audit_report": [
            (r"audit\s*report", 5),
            (r"independent\s*auditor", 4),
            (r"\bauditor'?s?\s*opinion\b", 4),
            (r"تقرير\s*المراجع[هة]?", 5),
            (r"مراجع\s*حسابات", 4),
            (r"راي\s*المراجع|رأي\s*المراجع", 4),
        ],
        # Payment / interim certificate — high weights so it beats invoice vocabulary.
        "payment_certificate": [
            (r"payment\s*certificate", 6),
            (r"interim\s*payment\s*certificate|\bipc\b", 6),
            (r"gross\s*(certified|valuation)|amount\s*certified|certified\s*(gross|amount|payable)", 5),
            (r"less\s*retention|retention\s*held|previously\s*certified", 4),
            (r"work\s*executed\s*to\s*date|value\s*of\s*work\s*done", 3),
            (r"شهاد[ةه]\s*(استحقاق\s*)?(ال)?دفع|مستخلص", 6),
            (r"المبلغ\s*المعتمد|القيم[ةه]\s*المعتمد[ةه]", 4),
        ],
        # QS measurement report — would score as a contract without its own bucket.
        "qs_report": [
            (r"quantity\s*survey(or|ing)?|\bqs\s*report|certification\s*report", 6),
            (r"measured\s*quantit|bill\s*of\s*quantities|\bboq\b|take[-\s]*off", 4),
            (r"مساح[ةه]\s*(ال)?كميات|تقرير\s*(ال)?كميات|حصر\s*(ال)?كميات", 6),
        ],
        # Board minutes / resolution approving a payment or variation.
        "board_resolution": [
            (r"board\s*(of\s*directors\s*)?(meeting\s*)?(minutes|resolution)", 6),
            (r"\bresolved\s*(that|to)\b|quorum|\bmotion\b|unanimously", 3),
            (r"محضر\s*اجتماع|قرار\s*مجلس\s*(ال)?[إا]دار[ةه]|مجلس\s*(ال)?[إا]دار[ةه]", 5),
        ],
    }

    scores = {k: 0 for k in patterns}

    # Score both content and filename; filename is a strong hint for scanned docs.
    for doc_type, rules in patterns.items():
        for pattern, weight in rules:
            # Frequency-weighted, capped.
            hits = len(re.findall(pattern, text_lower, flags=re.IGNORECASE))
            if hits:
                scores[doc_type] += weight * min(hits, 3)
            if re.search(pattern, filename_lower, flags=re.IGNORECASE):
                scores[doc_type] += max(1, weight // 2)

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Below threshold stays unknown.
    if best_score >= 5:
        return best_type

    # Financial fallback for OCR-noisy invoices/statements.
    amount_like = len(re.findall(r"(?:\d[\d,\.]{2,}\s*(?:egp|usd|eur|sar|aed|ج\.م|جنيه)?)", text_lower))
    finance_terms = len(
        re.findall(
            r"(amount|total|vat|tax|payment|invoice|statement|transaction|balance|رصيد|المبلغ|فاتور[هة]|كشف\s*حساب)",
            text_lower,
            flags=re.IGNORECASE,
        )
    )
    if amount_like >= 5 and finance_terms >= 4:
        if re.search(r"statement|transaction|debit|credit|كشف\s*حساب|رصيد", text_lower, re.IGNORECASE):
            return "bank_statement"
        return "invoice"

    return "unknown"


# ── Table-aware text extraction ────────────────────────────────────────────────
# Plain PyMuPDF text emits one table cell per line, so rows split across chunks and
# the LLM mis-pairs refs with amounts. Tables are lifted as one `a | b | c` line per row.

_TABLE_MIN_COLS = 3
_HEADER_HINT_RE = re.compile(
    r"date|ref|txn|description|debit|credit|balance|amount|qty|quantity|rate|item"
    r"|تاريخ|مرجع|بيان|مدين|دائن|رصيد|مبلغ|كمية",
    re.IGNORECASE,
)
_ROW_MARKER = "[TABLE ROW]"


def _clean_cell(cell) -> str:
    return re.sub(r"\s+", " ", str(cell or "")).strip()


def _looks_like_header(cells: list[str]) -> bool:
    """A header row names columns and carries no amounts of its own."""
    joined = " ".join(cells)
    has_hint = bool(_HEADER_HINT_RE.search(joined))
    has_amount = bool(re.search(r"\d[\d,]{3,}", joined))
    return has_hint and not has_amount


def _extract_page_tables(page: fitz.Page) -> tuple[list[fitz.Rect], list[tuple[str, list[str]]]]:
    """(table_rects, [(header, rows)]) per page. Header carries forward across
    fragments; fragments narrower than _TABLE_MIN_COLS (footers) are ignored."""
    try:
        found = page.find_tables()
    except Exception as e:  # find_tables is best-effort; never fail ingestion on it
        logger.debug("find_tables failed on page: %s", e)
        return [], []

    rects: list[fitz.Rect] = []
    blocks: list[tuple[str, list[str]]] = []
    carried_header: str | None = None
    for table in found.tables:
        if (table.col_count or 0) < _TABLE_MIN_COLS:
            continue
        rows = [[_clean_cell(c) for c in r] for r in table.extract()]
        rows = [r for r in rows if any(r)]
        if not rows:
            continue
        header = carried_header
        if _looks_like_header(rows[0]):
            header = " | ".join(rows[0])
            carried_header = header
            rows = rows[1:]
        rects.append(fitz.Rect(table.bbox))
        if not rows:
            continue  # header-only fragment: still masks its region, defines header
        blocks.append((header or "", [" | ".join(r) for r in rows]))
    return rects, blocks


def _text_outside_rects(page: fitz.Page, rects: list[fitz.Rect]) -> str:
    """Page prose with every text block that overlaps a lifted table removed."""
    if not rects:
        return page.get_text("text")
    parts: list[str] = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        if len(block) > 6 and block[6] != 0:  # image block
            continue
        if any(fitz.Rect(x0, y0, x1, y1).intersects(r) for r in rects):
            continue
        parts.append(text)
    return "\n".join(parts)


def _chunk_table_rows(header: str, rows: list[str], limit: int = _CHUNK_SIZE) -> list[str]:
    """Group rows into chunks under `limit` chars; rows are never split; header repeats."""
    prefix = f"{_ROW_MARKER} columns: {header}" if header else _ROW_MARKER
    chunks: list[str] = []
    current: list[str] = []
    size = len(prefix)
    for row in rows:
        line = f"{_ROW_MARKER} {row}"
        if current and size + len(line) + 1 > limit:
            chunks.append("\n".join([prefix, *current]))
            current, size = [], len(prefix)
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join([prefix, *current]))
    return chunks


def process_pdf(
    file_bytes: bytes,
    filename: str,
    doc_id: Optional[str] = None,
) -> tuple[DocumentMeta, list[DocumentChunk]]:
    """
    Full pipeline: PDF bytes → DocumentMeta + list[DocumentChunk].
    Chunks are NOT stored in Qdrant here — caller decides when to store.
    """
    if doc_id is None:
        doc_id = str(uuid.uuid4())

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_count = len(doc)

    all_text_by_page: list[tuple[int, str, bool]] = []  # (page_num, text, ocr_used)
    tables_by_page: dict[int, list[tuple[str, list[str]]]] = {}
    ocr_used = False

    for page_num in range(page_count):
        page = doc[page_num]
        text = page.get_text("text")

        if _is_scanned_page(text):
            text = _ocr_page(page)
            ocr_used = True
            used_ocr = True
        else:
            used_ocr = False
            # Lift tables as whole rows; drop their flattened cells from the prose.
            table_rects, table_blocks = _extract_page_tables(page)
            if table_blocks:
                text = _text_outside_rects(page, table_rects)
                tables_by_page[page_num + 1] = table_blocks

        if text.strip() or tables_by_page.get(page_num + 1):
            all_text_by_page.append((page_num + 1, text, used_ocr))

    doc.close()

    # Full text (table rows included) for type/language detection.
    table_text = "\n".join(
        "\n".join([hdr, *rows]) for blocks in tables_by_page.values() for hdr, rows in blocks
    )
    full_text = "\n".join(t for _, t, _ in all_text_by_page) + ("\n" + table_text if table_text else "")
    doc_language = detect_language(full_text)
    doc_type = _classify_document_type(full_text, filename)

    # Extract summary metadata
    from app.utils.arabic_normalizer import extract_amounts
    amounts_found = [a["raw"] for a in extract_amounts(full_text)][:10]

    meta = DocumentMeta(
        doc_id=doc_id,
        filename=filename,
        doc_type=doc_type,
        language=doc_language,
        page_count=page_count,
        amounts_found=amounts_found,
        ocr_used=ocr_used,
    )

    # Build chunks preserving page metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "،", " ", ""],
    )

    chunks: list[DocumentChunk] = []
    chunk_index = 0

    def _add_chunk(page_num: int, chunk_text: str) -> None:
        nonlocal chunk_index
        if len(chunk_text.strip()) < 20:
            return
        chunks.append(
            DocumentChunk(
                doc_id=doc_id,
                # Keep the full chunk text so downstream checks can inspect
                # all extracted schedule/payment lines without truncation.
                text=chunk_text,
                normalized_text=chunk_text,
                page_num=page_num,
                chunk_index=chunk_index,
                language=detect_language(chunk_text),
            )
        )
        chunk_index += 1

    for page_num, raw_text, _ in all_text_by_page:
        normalized = normalize_mixed_text(raw_text)
        for chunk_text in splitter.split_text(normalized):
            _add_chunk(page_num, chunk_text)
        # Table rows chunk separately: never split, header repeated per chunk.
        for header, rows in tables_by_page.get(page_num, []):
            for chunk_text in _chunk_table_rows(
                normalize_mixed_text(header), [normalize_mixed_text(r) for r in rows]
            ):
                _add_chunk(page_num, chunk_text)

    logger.info(
        "Processed '%s': %d pages, %d chunks, type=%s, lang=%s, ocr=%s",
        filename, page_count, len(chunks), doc_type, doc_language, ocr_used,
    )

    return meta, chunks


def ingest_document(file_bytes: bytes, filename: str, doc_id: Optional[str] = None) -> DocumentMeta:
    """Full ingestion: process PDF and store chunks in Qdrant."""
    meta, chunks = process_pdf(file_bytes, filename, doc_id)
    stored = store_chunks(chunks)
    logger.info("Ingested '%s': stored %d chunks in Qdrant", filename, stored)
    return meta
