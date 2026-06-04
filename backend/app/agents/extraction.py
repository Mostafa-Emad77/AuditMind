"""Extraction Agent — first node in the LangGraph audit pipeline."""
import asyncio
import hashlib
import logging

from langgraph.config import get_stream_writer
from langchain_core.messages import AIMessage

from app.models.state import AuditState
from app.models.schemas import ReasoningStep, DocumentChunk, DocumentMeta
from app.services.entity_extractor import extract_entities_from_chunks
from app.services.vector_store import semantic_search
from app.services.graph_builder import (
    store_document_node,
    store_entities,
    store_relationships,
)
from app.config import get_settings

logger = logging.getLogger(__name__)

# Per-doc-type targeted retrieval queries that complement the broad base query.
_DOC_TYPE_QUERIES: dict[str, list[str]] = {
    "invoice": [
        "amount date party name contract invoice",
        "invoice total subtotal VAT tax line items",
        "invoice number due date vendor client",
    ],
    "contract": [
        "amount date party name contract invoice",
        "contract value payment schedule milestones",
        "parties signatures clauses obligations",
    ],
    "bank_statement": [
        "amount date party name contract invoice",
        "bank transaction debit credit balance",
        "payment transfer beneficiary reference",
    ],
    "balance_sheet": [
        "amount date party name contract invoice",
        "assets liabilities equity total",
        "balance sheet entries accounts",
    ],
    "audit_report": [
        "amount date party name contract invoice",
        "audit findings figures cited documents",
        "audit clauses recommendations",
    ],
}
_BASE_QUERIES = [
    "amount date party name contract invoice",
    "payment schedule milestone value total",
]


def _adaptive_top_k(doc: DocumentMeta, base_k: int, max_k: int) -> int:
    """Scale top_k with page count, bounded by max_k."""
    pages = getattr(doc, "page_count", 1) or 1
    if pages <= 5:
        return base_k
    if pages <= 15:
        return min(base_k + 15, max_k)
    return max_k


def _chunk_key(r: dict) -> str:
    """Stable dedup key: (doc_id, page_num, text[:120] hash)."""
    text_fragment = (r.get("text") or "")[:120]
    raw = f"{r.get('doc_id','')}-{r.get('page_num', 0)}-{text_fragment}"
    return hashlib.md5(raw.encode("utf-8", errors="replace")).hexdigest()


def _retrieve_chunks_for_doc(doc: DocumentMeta, base_k: int, max_k: int) -> list[dict]:
    """
    Multi-query adaptive retrieval for a single document.
    Runs 2-3 targeted queries (depending on doc type), deduplicates by
    (doc_id, page_num, text prefix hash), and caps at max_k unique chunks.
    """
    queries = _DOC_TYPE_QUERIES.get(doc.doc_type, _BASE_QUERIES)
    per_query_k = _adaptive_top_k(doc, base_k, max_k)

    seen: dict[str, dict] = {}
    for query in queries:
        if len(seen) >= max_k:
            break
        results = semantic_search(query=query, doc_ids=[doc.doc_id], top_k=per_query_k)
        for r in results:
            if len(seen) >= max_k:
                break
            key = _chunk_key(r)
            if key not in seen:
                seen[key] = r

    return list(seen.values())


def _emit(writer, agent: str, step_type: str, content: str, **kwargs) -> ReasoningStep:
    """Emit a reasoning step via the stream writer and return it."""
    step = ReasoningStep(
        agent=agent,
        step_type=step_type,
        content=content,
        **kwargs,
    )
    writer({"type": "reasoning_step", "step": step.model_dump(mode="json")})
    return step


async def extraction_agent(state: AuditState) -> dict:
    """
    Extraction Agent Node.
    Reads document chunks from Qdrant (already stored during upload),
    extracts entities/relationships, and builds the Neo4j knowledge graph.
    """
    writer = get_stream_writer()
    new_steps: list[ReasoningStep] = []
    documents = state["documents"]

    _emit(writer, "extraction", "thought",
          f"Starting extraction phase. I have {len(documents)} document(s) to analyze.")

    if not documents:
        _emit(writer, "extraction", "summary", "No documents to process.")
        return {
            "messages": [AIMessage(content="No documents provided.")],
            "extraction_complete": False,
            "reasoning_trace": new_steps,
        }

    # List what we have
    doc_summary_parts = []
    for doc in documents:
        lang_label = {"arabic": "Arabic", "english": "English", "mixed": "Mixed Arabic/English"}.get(doc.language, "Unknown language")
        doc_summary_parts.append(
            f"- {doc.filename}: {lang_label} {doc.doc_type.replace('_', ' ')}, {doc.page_count} page(s)"
            + (" [OCR used]" if doc.ocr_used else "")
        )

    overview = "Documents received:\n" + "\n".join(doc_summary_parts)
    step = _emit(writer, "extraction", "thought", overview)
    new_steps.append(step)

    # Neo4j schema is initialized once at app startup (see main.py lifespan).
    step = _emit(writer, "extraction", "thought", "Building knowledge graph from documents...")
    new_steps.append(step)

    all_entities = []
    all_relationships = []

    settings = get_settings()
    loop = asyncio.get_running_loop()

    async def _process_doc(doc: DocumentMeta) -> tuple[list, list]:
        """Retrieve chunks, extract entities/relationships, and persist to Neo4j for one document.

        Qdrant retrieval and Neo4j writes are synchronous network I/O, so they run in the
        default executor to keep the event loop free. Documents are independent, so callers
        gather these coroutines to process them concurrently.
        """
        step = _emit(writer, "extraction", "tool_call",
                     f"Extracting entities from: {doc.filename}",
                     tool_name="extract_entities",
                     tool_input={"doc_id": doc.doc_id, "doc_type": doc.doc_type})
        new_steps.append(step)

        # Adaptive multi-query retrieval from Qdrant (sync I/O → offload)
        chunks_raw = await loop.run_in_executor(
            None,
            _retrieve_chunks_for_doc,
            doc,
            settings.extraction_top_k_base,
            settings.extraction_top_k_max,
        )

        # Reconstruct minimal DocumentChunk objects for entity extraction
        chunks = [
            DocumentChunk(
                doc_id=doc.doc_id,
                text=r["text"] or "",
                normalized_text=r["text"] or "",
                page_num=r["page_num"] or 1,
                chunk_index=i,
                language=r["language"] or doc.language,
            )
            for i, r in enumerate(chunks_raw)
        ]

        logger.info(
            "Extraction retrieval — %s: %d unique chunks (base_k=%d, max_k=%d, pages=%d)",
            doc.filename,
            len(chunks),
            settings.extraction_top_k_base,
            settings.extraction_top_k_max,
            doc.page_count or 1,
        )

        step = _emit(
            writer,
            "extraction",
            "thought",
            f"Retrieved {len(chunks)} unique candidate chunks for {doc.filename} "
            f"(adaptive multi-query, max_k={settings.extraction_top_k_max}). "
            "Starting entity extraction...",
        )
        new_steps.append(step)

        async def _on_progress(
            processed: int,
            total: int,
            entities_so_far: int,
            relationships_so_far: int,
            chunk: DocumentChunk,
        ) -> None:
            # Throttle: emit only every 10th chunk (and the final one) to avoid
            # flooding the SSE stream on large documents.
            if processed % 10 != 0 and processed != total:
                return
            progress_step = _emit(
                writer,
                "extraction",
                "thought",
                (
                    f"Extraction progress {processed}/{total} for {doc.filename} "
                    f"(page {chunk.page_num}) — {entities_so_far} entities, "
                    f"{relationships_so_far} relationships so far."
                ),
            )
            new_steps.append(progress_step)

        try:
            entities, relationships = await extract_entities_from_chunks(
                chunks,
                doc_type=doc.doc_type,
                progress_callback=_on_progress,
            )

            step = _emit(writer, "extraction", "tool_result",
                         f"Found {len(entities)} entities and {len(relationships)} relationships in {doc.filename}",
                         tool_name="extract_entities",
                         tool_output=f"{len(entities)} entities, {len(relationships)} relationships")
            new_steps.append(step)

            # Store in Neo4j (sync driver → offload so it doesn't block the event loop)
            await loop.run_in_executor(None, store_document_node, doc)
            await loop.run_in_executor(None, store_entities, entities)
            await loop.run_in_executor(None, store_relationships, relationships)

            return entities, relationships

        except Exception as e:
            logger.error("Entity extraction failed for %s: %s", doc.filename, e)
            step = _emit(writer, "extraction", "thought",
                         f"Entity extraction encountered an issue for {doc.filename}: {e}")
            new_steps.append(step)
            return [], []

    # Documents are independent → process them concurrently.
    results = await asyncio.gather(*[_process_doc(doc) for doc in documents])
    for entities, relationships in results:
        all_entities.extend(entities)
        all_relationships.extend(relationships)

    # Summary
    summary = (
        f"Extraction complete. "
        f"Total: {len(all_entities)} entities and {len(all_relationships)} relationships "
        f"stored in Neo4j knowledge graph across {len(documents)} document(s)."
    )
    step = _emit(writer, "extraction", "summary", summary)
    new_steps.append(step)

    return {
        "messages": [AIMessage(content=summary)],
        "extraction_complete": True,
        "reasoning_trace": new_steps,
    }
