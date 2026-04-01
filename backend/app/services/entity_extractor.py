"""LLM-based entity and relationship extraction from document chunks."""
import asyncio
import json
import logging
from typing import Optional, Callable, Awaitable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.models.schemas import Entity, Relationship, DocumentChunk
from app.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial document analysis expert specializing in Arabic and English documents.
Extract all entities and relationships from the provided text chunk.

Return a JSON object with exactly this structure:
{{
  "entities": [
    {{
      "entity_type": "person|company|amount|date|contract_id|invoice_id|clause|other",
      "value": "original extracted text",
      "normalized_value": "cleaned/normalized version",
      "source_language": "arabic|english|mixed|unknown — language of the surface text for THIS entity span",
      "coreference_note": "null or short text: if the same real-world concept appears in Arabic in one place and English elsewhere in this chunk, state the link explicitly (e.g. 'same party as …')",
      "amount_role": "(only for amount entities) one of: total_contract_value|total_invoice|single_payment|opening_balance|closing_balance|milestone_scheduled|retainer|vat_tax|unknown",
      "western_numeral_form": "(only for amount entities, else null) the amount written with Western digits 0-9 if present",
      "arabic_indic_numeral_form": "(only for amount entities, else null) the amount written with Arabic-Indic digits ١٢٣ if present",
      "numeral_mismatch": "(only for amount entities, else null) true if both forms above are present and denote different numeric values; false if both present and agree; null if only one script appears"
    }}
  ],
  "relationships": [
    {{
      "source_value": "entity value 1",
      "target_value": "entity value 2",
      "relationship_type": "signed_by|references|payment_for|total_value_of|amount_of|dated|party_to|issued_by|opening_balance_of|milestone_amount_of|other"
    }}
  ]
}}

Rules:
- Extract person names, company names, monetary amounts (with currency), dates, contract/invoice IDs
- SOURCE LANGUAGE: For every entity, set "source_language" to the language of the text span you extracted (arabic / english / mixed / unknown). This is required for downstream contradiction checks.
- ARABIC-INDIC VS WESTERN DIGITS (١٢٣ vs 123): The same line may show a value in one script, both, or ambiguously. For each amount entity, fill western_numeral_form and arabic_indic_numeral_form when those digit forms appear; set numeral_mismatch to true only when both forms are present and interpret to different values. If only one script appears, leave the other null and numeral_mismatch null. Always put a canonical Western-digit form with currency in normalized_value for amounts when you can determine it.
- LEGAL ENTITY SUFFIXES: Suffixes such as Co., LLC, S.A.E., Trading, Ltd., ذ.م.م are semantically significant. Never merge or normalize two company names into one if the legal form / suffix differs (e.g. "Delta Trading" vs "Delta Trading S.A.E." are distinct entities). Preserve the full legal designation in normalized_value for companies.
- BILINGUAL COREFERENCE: The same obligation, party, or amount may be named in Arabic in one clause and English in another. When you infer within this chunk that two mentions refer to the same real-world thing, record it in coreference_note for at least one of the entities; do not silently collapse distinct surface forms without stating the link.
- For monetary amounts, always include the currency code in normalized_value (e.g., "50000 EGP")
- For amount entities, you MUST set amount_role to classify the semantic purpose:
  * total_contract_value — the full agreed contract price / total value
  * total_invoice — the invoice grand total or total due
  * single_payment — an individual bank transaction, wire transfer, or payment
  * opening_balance — a bank statement opening / beginning balance
  * closing_balance — a bank statement closing / ending balance
  * milestone_scheduled — a scheduled installment or milestone payment in a contract schedule
  * retainer — a monthly retainer fee in a contract schedule
  * vat_tax — a VAT or tax amount
  * unknown — when the role cannot be determined
- For relationship_type, use richer types to distinguish:
  * total_value_of — links a total amount to its parent (contract, invoice)
  * payment_for — links a bank payment to the entity it pays
  * opening_balance_of — links an opening balance to its bank account
  * milestone_amount_of — links a scheduled milestone amount to its contract
- Dates: normalize to YYYY-MM-DD format if possible
- Only extract relationships you are confident about
- Return valid JSON only, no explanations"""),
    ("human", "Document type: {doc_type}\nLanguage: {language}\n\nText:\n{text}")
])


async def extract_entities_from_chunk(
    chunk: DocumentChunk,
    doc_type: str = "unknown",
    max_retries: int = 2,
) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities and relationships from a single document chunk (async)."""
    llm = get_llm(temperature=0.0)
    parser = JsonOutputParser()
    chain = _EXTRACTION_PROMPT | llm | parser

    for attempt in range(max_retries + 1):
        try:
            result = await chain.ainvoke({
                "doc_type": doc_type,
                "language": chunk.language,
                "text": chunk.normalized_text[:2000],
            })

            entities = []
            _lang_ok = frozenset({"arabic", "english", "mixed", "unknown"})
            for e in result.get("entities", []):
                try:
                    etype = e.get("entity_type", "other")
                    raw_lang = str(e.get("source_language") or "unknown").strip().lower()
                    if raw_lang not in _lang_ok:
                        raw_lang = "unknown"
                    kwargs: dict = dict(
                        entity_type=etype,
                        value=e.get("value", ""),
                        normalized_value=e.get("normalized_value", e.get("value", "")),
                        source_doc_id=chunk.doc_id,
                        source_page=chunk.page_num,
                        confidence=0.9,
                        source_language=raw_lang,
                        coreference_note=(str(e.get("coreference_note")).strip() or None)
                        if e.get("coreference_note") is not None
                        else None,
                        western_numeral_form=(str(e.get("western_numeral_form")).strip() or None)
                        if e.get("western_numeral_form") is not None
                        else None,
                        arabic_indic_numeral_form=(
                            str(e.get("arabic_indic_numeral_form")).strip() or None
                        )
                        if e.get("arabic_indic_numeral_form") is not None
                        else None,
                    )
                    nm_raw = e.get("numeral_mismatch")
                    if nm_raw is None:
                        kwargs["numeral_mismatch"] = None
                    elif isinstance(nm_raw, bool):
                        kwargs["numeral_mismatch"] = nm_raw
                    else:
                        kwargs["numeral_mismatch"] = str(nm_raw).lower() in ("true", "1", "yes")
                    if kwargs.get("coreference_note") == "":
                        kwargs["coreference_note"] = None
                    if etype == "amount":
                        raw_role = (e.get("amount_role") or "unknown").strip().lower()
                        _valid_roles = {
                            "total_contract_value", "total_invoice",
                            "single_payment", "opening_balance",
                            "closing_balance", "milestone_scheduled",
                            "retainer", "vat_tax", "unknown",
                        }
                        kwargs["amount_role"] = raw_role if raw_role in _valid_roles else "unknown"
                    else:
                        kwargs["western_numeral_form"] = None
                        kwargs["arabic_indic_numeral_form"] = None
                        kwargs["numeral_mismatch"] = None
                    entities.append(Entity(**kwargs))
                except Exception:
                    continue

            relationships = []
            entity_map = {e.value: e for e in entities}
            entity_map.update({e.normalized_value: e for e in entities})

            for r in result.get("relationships", []):
                src = entity_map.get(r.get("source_value", ""))
                tgt = entity_map.get(r.get("target_value", ""))
                if src and tgt:
                    relationships.append(Relationship(
                        source_entity_id=src.entity_id,
                        target_entity_id=tgt.entity_id,
                        relationship_type=r.get("relationship_type", "other"),
                        source_doc_id=chunk.doc_id,
                        source_page=chunk.page_num,
                        confidence=0.85,
                    ))

            return entities, relationships

        except Exception as e:
            if attempt == max_retries:
                logger.warning(
                    "Entity extraction failed for chunk %s after %d attempts: %s",
                    chunk.chunk_id, max_retries + 1, e,
                )
                return [], []
            logger.debug("Retry %d for chunk %s: %s", attempt + 1, chunk.chunk_id, e)

    return [], []


async def extract_entities_from_chunks(
    chunks: list[DocumentChunk],
    doc_type: str = "unknown",
    max_chunks: int | None = None,
    progress_callback: Optional[
        Callable[[int, int, int, int, DocumentChunk], Awaitable[None] | None]
    ] = None,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Extract entities from multiple chunks sequentially with rate-limit pacing.

    max_chunks defaults to None, in which case it is calculated dynamically:
      - 5 chunks per page of the source document, capped at 15.
    This means a 2-page CV gets ~10 chunks, a 24-page report gets 15.

    Requests are sent one-at-a-time with a 4-second gap so we stay well under
    the OpenRouter free-tier limit of 20 req/min (= 1 req per 3 seconds).
    """
    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []

    meaningful = [c for c in chunks if len(c.normalized_text) >= 100]

    # Dynamic limit: 5 chunks per page, min 5, max 15
    if max_chunks is None:
        page_count = max((c.page_num for c in chunks), default=1)
        max_chunks = min(max(page_count * 5, 5), 15)

    to_process = meaningful[:max_chunks]
    total = len(to_process)
    logger.info("Entity extraction: processing %d/%d meaningful chunks", total, len(meaningful))

    # Sequential with 4-second pacing — 15 req/min, safely under 20 req/min limit
    for i, chunk in enumerate(to_process):
        if i > 0:
            await asyncio.sleep(4)

        try:
            entities, relationships = await extract_entities_from_chunk(chunk, doc_type)
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        except Exception as e:
            logger.warning("Extraction error for chunk %s: %s", chunk.chunk_id, e)

        if progress_callback is not None:
            try:
                maybe_awaitable = progress_callback(
                    i + 1,
                    total,
                    len(all_entities),
                    len(all_relationships),
                    chunk,
                )
                if asyncio.iscoroutine(maybe_awaitable):
                    await maybe_awaitable
            except Exception as e:
                logger.debug("Progress callback failed: %s", e)

        if (i + 1) % 5 == 0 or (i + 1) == total:
            logger.info("Extracted entities from %d/%d chunks", i + 1, total)

    logger.info(
        "Total extracted: %d entities, %d relationships from %d chunks",
        len(all_entities), len(all_relationships), total,
    )
    return all_entities, all_relationships
