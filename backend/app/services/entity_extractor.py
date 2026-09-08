"""LLM-based entity and relationship extraction from document chunks."""
import asyncio
import logging
from typing import Optional, Callable, Awaitable, get_args

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.models.schemas import AMOUNT_ROLES, Entity, Relationship, DocumentChunk
from app.utils.llm_factory import get_llm
from app.utils.canonical_id import canonical_entity_id, canonical_rel_id
from app.utils.money_parse import parse_monetary_amount, extract_currency_code
from app.config import get_settings

logger = logging.getLogger(__name__)

# Derived from the schema so the accepted set can never drift from the Literal.
_VALID_AMOUNT_ROLES: frozenset[str] = frozenset(get_args(AMOUNT_ROLES))
_VALID_AUTHORITY_LEVELS: frozenset[str] = frozenset(
    {"chairman", "ceo", "cfo", "director", "manager", "other", "unknown"}
)

# Title keywords → authority level, most senior first; backstops the LLM's level.
_AUTHORITY_FROM_TITLE: tuple[tuple[str, str], ...] = (
    ("chairman", "chairman"),
    ("رئيس مجلس", "chairman"),
    ("chief executive", "ceo"),
    ("ceo", "ceo"),
    ("المدير التنفيذي", "ceo"),
    ("chief financial", "cfo"),
    ("cfo", "cfo"),
    ("finance director", "director"),
    ("managing director", "director"),
    ("director", "director"),
    ("مدير عام", "director"),
    ("manager", "manager"),
    ("مدير", "manager"),
)


def _authority_from_title(title: Optional[str]) -> Optional[str]:
    """Infer a signing authority level from a stated job title."""
    if not title:
        return None
    t = title.strip().lower()
    for keyword, level in _AUTHORITY_FROM_TITLE:
        if keyword in t:
            return level
    return None

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
      "amount_role": "(only for amount entities) one of: total_contract_value|milestone_scheduled|retainer|total_invoice|invoice_subtotal|invoice_line_item|invoice_referenced_contract_value|transaction_debit|transaction_credit|statement_total_debits|statement_total_credits|running_balance|opening_balance|closing_balance|vat_tax|late_fee|unknown",
      "transaction_ref": "(only for amount entities on a bank-statement transaction row, else null) the row's Reference / TXN number exactly as written, e.g. 'TXN-Q1-010'",
      "transaction_date": "(only for amount entities on a bank-statement transaction row, else null) that row's date, normalized to YYYY-MM-DD if possible",
      "western_numeral_form": "(only for amount entities, else null) the amount written with Western digits 0-9 if present",
      "arabic_indic_numeral_form": "(only for amount entities, else null) the amount written with Arabic-Indic digits ١٢٣ if present",
      "numeral_mismatch": "(only for amount entities, else null) true if both forms above are present and denote different numeric values; false if both present and agree; null if only one script appears",
      "role_title": "(only for person entities, else null) the person's stated title exactly as written, e.g. 'Chairman', 'CFO', 'CEO - Nile Financial Consulting'",
      "signing_authority_level": "(only for person entities, else null) one of: chairman|ceo|cfo|director|manager|other|unknown",
      "document_signed": "(only for person entities, else null) what this person signed in THIS chunk, e.g. 'contract CTR-2024-044' or 'amendment 1'; null if they are merely mentioned, not signing"
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
  CONTRACT-SIDE:
  * total_contract_value — the full agreed contract price stated BY THE CONTRACT ITSELF
  * milestone_scheduled — a scheduled installment / milestone payment in a contract payment schedule
  * retainer — a recurring (e.g. monthly) retainer fee
  INVOICE-SIDE:
  * total_invoice — the invoice grand total / total due (after VAT)
  * invoice_subtotal — the invoice subtotal BEFORE VAT
  * invoice_line_item — a single line on an invoice that has no more specific role below.
    ROLE PRECEDENCE: a role describes what an amount IS, not where it sits. A recurring
    retainer billed as an invoice line is `retainer`, and a contract milestone billed as
    an invoice line is `milestone_scheduled` — in both cases use the specific role, NOT
    invoice_line_item. Getting this wrong hides rate changes: a retainer billed at a
    different rate than the contract sets can only be caught when both figures carry the
    `retainer` role.
  * invoice_referenced_contract_value — an invoice RESTATING what it claims the contract's
    total value is (e.g. "against Contract CTR-2024-044, value EGP 180,000"). This is NOT
    total_contract_value. Use this role whenever a NON-contract document asserts the
    contract's value — comparing the two is precisely how restatement errors are caught,
    so they must never be given the same role.
  BANK-STATEMENT-SIDE (read the column headers carefully):
  * transaction_debit — money OUT of the account on one transaction row (Debit column)
  * transaction_credit — money IN to the account on one transaction row (Credit column)
  * running_balance — the per-row Balance column showing the account balance AFTER that row
  * statement_total_debits — the statement's own stated "Total Debits" summary figure
  * statement_total_credits — the statement's own stated "Total Credits" summary figure
  * opening_balance — the statement's opening / beginning balance
  * closing_balance — the statement's closing / ending balance
  OTHER:
  * vat_tax — a VAT or tax amount
  * late_fee — a late-payment penalty or interest charge
  * unknown — ONLY when the role genuinely cannot be determined from context
- BANK STATEMENT ROWS — three DISTINCT amounts per transaction row, NEVER merged:
  A transaction row typically reads: Date | Ref | Description | Debit | Credit | Balance.
  Emit a SEPARATE amount entity for each populated column, with its own role:
  the Debit figure as transaction_debit, the Credit figure as transaction_credit,
  and the Balance figure as running_balance. Never emit one merged "amount" for a row,
  and never label a Balance figure as a debit or credit.
- [TABLE ROW] LINES: a line starting with "[TABLE ROW]" is ONE complete table row with
  cells separated by " | ", in the column order given on the "[TABLE ROW] columns:" line.
  Read each such line as a single record: take the ref, date, debit, credit and balance
  from THAT line only. Never combine cells from two different [TABLE ROW] lines.
- TRANSACTION IDENTITY: for every bank-statement transaction amount (debit or credit),
  also fill transaction_ref (the row's Reference / TXN number) and transaction_date (the
  row's date). Two rows are the SAME transaction only if ref, date AND amount all match;
  a shared payee/beneficiary name alone NEVER makes two rows the same transaction.
- RUNNING BALANCE IS NOT A TRANSACTION: running_balance values are cumulative account
  state, not transaction values. They must never be summed, never compared to invoice or
  contract totals, and never used in any "total paid" calculation. Tagging a Balance-column
  figure as anything other than running_balance corrupts every downstream total, so when a
  number is a balance, always say so.
- PERSON / SIGNATORY ENTITIES: for every named person, set role_title and
  signing_authority_level from their stated title. Set document_signed only when the chunk
  shows them actually signing/approving something. This is what enables authority-mismatch
  detection (e.g. an amendment signed by a CFO where the original was signed by the Chairman).
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
    llm = get_llm(temperature=0.0, role="ner_arabic")
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
                    norm_val = e.get("normalized_value", e.get("value", ""))
                    raw_val = e.get("value", "")
                    # Deterministic amount parsing — independent of LLM string formatting.
                    parsed_amount: float | None = None
                    parsed_currency: str | None = None
                    txn_ref: str | None = None
                    txn_date: str | None = None
                    if etype == "amount":
                        # Prefer normalized_value (the LLM is instructed to put canonical Western form there).
                        for candidate in (norm_val, raw_val, e.get("western_numeral_form")):
                            if candidate and parsed_amount is None:
                                parsed_amount = parse_monetary_amount(candidate)
                            if candidate and parsed_currency is None:
                                parsed_currency = extract_currency_code(candidate)
                            if parsed_amount is not None and parsed_currency is not None:
                                break
                        txn_ref = (str(e.get("transaction_ref")).strip() or None) \
                            if e.get("transaction_ref") is not None else None
                        txn_date = (str(e.get("transaction_date")).strip() or None) \
                            if e.get("transaction_date") is not None else None
                    kwargs: dict = dict(
                        entity_id=canonical_entity_id(
                            etype, norm_val, chunk.doc_id,
                            amount_value=parsed_amount,
                            amount_currency=parsed_currency,
                            txn_ref=txn_ref,
                            txn_date=txn_date,
                        ),
                        entity_type=etype,
                        value=raw_val,
                        normalized_value=norm_val,
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
                        kwargs["amount_role"] = raw_role if raw_role in _VALID_AMOUNT_ROLES else "unknown"
                        kwargs["amount_value"] = parsed_amount
                        kwargs["amount_currency"] = parsed_currency
                        kwargs["txn_ref"] = txn_ref
                        kwargs["txn_date"] = txn_date
                    else:
                        kwargs["western_numeral_form"] = None
                        kwargs["arabic_indic_numeral_form"] = None
                        kwargs["numeral_mismatch"] = None
                        kwargs["amount_value"] = None
                        kwargs["amount_currency"] = None
                    if etype == "person":
                        raw_auth = str(e.get("signing_authority_level") or "").strip().lower()
                        title = (str(e.get("role_title")).strip() or None) \
                            if e.get("role_title") is not None else None
                        auth = raw_auth if raw_auth in _VALID_AUTHORITY_LEVELS else None
                        # Stated title beats a non-committal level from the model.
                        if auth in (None, "other", "unknown"):
                            auth = _authority_from_title(title) or auth
                        kwargs["signing_authority_level"] = auth
                        kwargs["role_title"] = title
                        kwargs["document_signed"] = (str(e.get("document_signed")).strip() or None) \
                            if e.get("document_signed") is not None else None
                    entities.append(Entity(**kwargs))
                except Exception:
                    continue

            # Dedupe within chunk by canonical id.
            entities = list({ent.entity_id: ent for ent in entities}.values())

            relationships = []
            entity_map: dict[str, Entity] = {}
            for ent in entities:
                # Lowercased keys absorb casing/whitespace drift.
                entity_map[ent.value.strip().lower()] = ent
                entity_map[ent.normalized_value.strip().lower()] = ent

            dropped = 0
            for r in result.get("relationships", []):
                src_key = (r.get("source_value") or "").strip().lower()
                tgt_key = (r.get("target_value") or "").strip().lower()
                src = entity_map.get(src_key)
                tgt = entity_map.get(tgt_key)
                if src and tgt and src.entity_id != tgt.entity_id:
                    rel_type = r.get("relationship_type", "other")
                    relationships.append(Relationship(
                        rel_id=canonical_rel_id(src.entity_id, tgt.entity_id, rel_type),
                        source_entity_id=src.entity_id,
                        target_entity_id=tgt.entity_id,
                        relationship_type=rel_type,
                        source_doc_id=chunk.doc_id,
                        source_page=chunk.page_num,
                        confidence=0.85,
                    ))
                else:
                    dropped += 1
            if dropped:
                logger.debug(
                    "Dropped %d relationship(s) in chunk %s (unresolved endpoints)",
                    dropped, chunk.chunk_id,
                )

            # Dedupe relationships within this chunk by canonical rel_id.
            relationships = list({rel.rel_id: rel for rel in relationships}.values())

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
    """Extract from chunks concurrently. Cap: explicit max_chunks, else
    settings.extraction_chunk_cap (None = none). Concurrency/sleep from settings."""
    settings = get_settings()
    concurrency = max(1, settings.extraction_concurrency)
    chunk_sleep = max(0.0, settings.extraction_chunk_sleep)

    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []

    # Shared mutable progress counters (safe: asyncio is single-threaded)
    completed_count = 0

    meaningful = [c for c in chunks if len(c.normalized_text) >= 100]

    if max_chunks is None:
        max_chunks = settings.extraction_chunk_cap  # may stay None → no cap

    to_process = meaningful if max_chunks is None else meaningful[:max_chunks]
    total = len(to_process)
    logger.info(
        "Entity extraction: processing %d/%d meaningful chunks (concurrency=%d, sleep=%.1fs)",
        total, len(meaningful), concurrency, chunk_sleep,
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def _process_chunk(
        idx: int, chunk: DocumentChunk
    ) -> tuple[int, list[Entity], list[Relationship]]:
        """Run extraction for one chunk under the semaphore."""
        async with semaphore:
            try:
                entities, relationships = await extract_entities_from_chunk(chunk, doc_type)
            except Exception as e:
                logger.warning("Extraction error for chunk %s: %s", chunk.chunk_id, e)
                entities, relationships = [], []
            return idx, entities, relationships

    # Process in batches so progress_callback and sleep work naturally across groups
    batch_size = concurrency
    for batch_start in range(0, total, batch_size):
        batch = list(enumerate(to_process[batch_start: batch_start + batch_size], start=batch_start))

        results = await asyncio.gather(*[_process_chunk(idx, chunk) for idx, chunk in batch])

        # Results arrive in completion order; sort by original index for deterministic callback order
        for idx, entities, relationships in sorted(results, key=lambda r: r[0]):
            all_entities.extend(entities)
            all_relationships.extend(relationships)
            completed_count += 1
            chunk = to_process[idx]

            if progress_callback is not None:
                try:
                    maybe_awaitable = progress_callback(
                        completed_count,
                        total,
                        len(all_entities),
                        len(all_relationships),
                        chunk,
                    )
                    if asyncio.iscoroutine(maybe_awaitable):
                        await maybe_awaitable
                except Exception as e:
                    logger.debug("Progress callback failed: %s", e)

        logger.info(
            "Extracted entities from %d/%d chunks",
            min(batch_start + batch_size, total),
            total,
        )

        # Inter-batch sleep (only relevant for free-tier pacing)
        if chunk_sleep > 0 and (batch_start + batch_size) < total:
            await asyncio.sleep(chunk_sleep)

    logger.info(
        "Total extracted: %d entities, %d relationships from %d chunks",
        len(all_entities), len(all_relationships), total,
    )
    return all_entities, all_relationships
