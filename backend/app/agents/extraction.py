"""Extraction Agent — first node in the LangGraph audit pipeline."""
import json
import logging
from datetime import datetime

from langgraph.config import get_stream_writer
from langchain_core.messages import HumanMessage, AIMessage

from app.models.state import AuditState
from app.models.schemas import ReasoningStep
from app.services.entity_extractor import extract_entities_from_chunks
from app.services.graph_builder import (
    init_graph_schema,
    store_document_node,
    store_entities,
    store_relationships,
)

logger = logging.getLogger(__name__)


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

    # Initialize Neo4j schema
    try:
        init_graph_schema()
        step = _emit(writer, "extraction", "thought", "Neo4j schema initialized. Building knowledge graph...")
        new_steps.append(step)
    except Exception as e:
        logger.warning("Neo4j schema init failed (non-fatal): %s", e)
        step = _emit(writer, "extraction", "thought",
                     f"Note: Neo4j connection issue ({e}). Graph features will be limited.")
        new_steps.append(step)

    all_entities = []
    all_relationships = []

    for doc in documents:
        step = _emit(writer, "extraction", "tool_call",
                     f"Extracting entities from: {doc.filename}",
                     tool_name="extract_entities",
                     tool_input={"doc_id": doc.doc_id, "doc_type": doc.doc_type})
        new_steps.append(step)

        # Retrieve chunks from Qdrant
        from app.services.vector_store import semantic_search
        chunks_raw = semantic_search(
            query="amount date party name contract invoice",
            doc_ids=[doc.doc_id],
            top_k=50,
        )

        # Reconstruct minimal DocumentChunk objects for entity extraction
        from app.models.schemas import DocumentChunk
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

        step = _emit(
            writer,
            "extraction",
            "thought",
            f"Retrieved {len(chunks)} candidate chunks for {doc.filename}. Starting chunk-by-chunk extraction...",
        )
        new_steps.append(step)

        async def _on_progress(
            processed: int,
            total: int,
            entities_so_far: int,
            relationships_so_far: int,
            chunk: DocumentChunk,
        ) -> None:
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
            all_entities.extend(entities)
            all_relationships.extend(relationships)

            step = _emit(writer, "extraction", "tool_result",
                         f"Found {len(entities)} entities and {len(relationships)} relationships in {doc.filename}",
                         tool_name="extract_entities",
                         tool_output=f"{len(entities)} entities, {len(relationships)} relationships")
            new_steps.append(step)

            # Store in Neo4j
            store_document_node(doc)
            store_entities(entities)
            store_relationships(relationships)

        except Exception as e:
            logger.error("Entity extraction failed for %s: %s", doc.filename, e)
            step = _emit(writer, "extraction", "thought",
                         f"Entity extraction encountered an issue for {doc.filename}: {e}")
            new_steps.append(step)

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
