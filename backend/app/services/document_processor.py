"""Document ingestion pipeline: PDF → OCR → normalize → chunk → Qdrant."""
import io
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.schemas import DocumentMeta, DocumentChunk
from app.utils.arabic_normalizer import (
    detect_language,
    normalize_mixed_text,
    is_arabic_text,
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

_DOC_TYPES = {"invoice", "contract", "balance_sheet", "bank_statement", "audit_report", "unknown"}


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


def _classify_document_type(text: str, filename: str) -> str:
    """
    Heuristic-first document classification from content + filename.

    The regex classifier only commits above its confidence threshold, so a non-"unknown"
    result is trustworthy — and free. The LLM call (seconds of perceived upload latency
    per document) runs only when the heuristic can't decide.
    """
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
    """
    OCR-tolerant deterministic fallback classifier from content + filename.
    """
    text_lower = text.lower()
    # Separators become spaces so \b-anchored patterns can match: without this,
    # "\bcontract\b" never matches "contract_CTR2024044.pdf" (an underscore is a word
    # character, so there is no boundary after "contract") and the filename hint — often
    # the strongest signal for a scanned document — silently contributes nothing.
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
    }

    scores = {k: 0 for k in patterns}

    # Score both content and filename; filename is a strong hint for scanned docs.
    for doc_type, rules in patterns.items():
        for pattern, weight in rules:
            # Frequency-weighted, with diminishing returns. Presence-only scoring made a
            # contract saying "contract" 12 times indistinguishable from one passing
            # mention, so a contract that merely referenced invoices and VAT could
            # outscore its own type. The cap stops one repeated term monopolising.
            hits = len(re.findall(pattern, text_lower, flags=re.IGNORECASE))
            if hits:
                scores[doc_type] += weight * min(hits, 3)
            if re.search(pattern, filename_lower, flags=re.IGNORECASE):
                scores[doc_type] += max(1, weight // 2)

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Confidence threshold. Low scores remain unknown to avoid wild misclassification.
    if best_score >= 5:
        return best_type

    # Financial fallback: many amounts + finance terms (useful for OCR-noisy invoices/statements).
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

        if text.strip():
            all_text_by_page.append((page_num + 1, text, used_ocr))

    doc.close()

    # Combine all text to detect document type and language
    full_text = "\n".join(t for _, t, _ in all_text_by_page)
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

    for page_num, raw_text, _ in all_text_by_page:
        normalized = normalize_mixed_text(raw_text)
        page_chunks = splitter.split_text(normalized)

        for chunk_text in page_chunks:
            if len(chunk_text.strip()) < 20:
                continue
            lang = detect_language(chunk_text)
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    # Keep the full chunk text so downstream checks can inspect
                    # all extracted schedule/payment lines without truncation.
                    text=chunk_text,
                    normalized_text=chunk_text,
                    page_num=page_num,
                    chunk_index=chunk_index,
                    language=lang,
                )
            )
            chunk_index += 1

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
