"""Qdrant vector store operations for AuditMind."""
import logging
from typing import Optional

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchText,
    PayloadSchemaType,
    TextIndexParams,
    TokenizerType,
)
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.models.schemas import DocumentChunk
from app.utils.canonical_id import canonical_chunk_point_id

logger = logging.getLogger(__name__)

_encoder: Optional[SentenceTransformer] = None


def get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        settings = get_settings()
        _encoder = SentenceTransformer(settings.embedding_model)
    return _encoder


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    kwargs: dict = {"url": settings.qdrant_url, "timeout": 120}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key
    return QdrantClient(**kwargs)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = 768) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection: %s", collection_name)
    # Qdrant Cloud requires payload indexes for filtered search.
    # Safe to call repeatedly; if index exists, Qdrant treats it as update/no-op.
    client.create_payload_index(
        collection_name=collection_name,
        field_name="doc_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="language",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    # Full-text index on chunk text — enables keyword_search() for exact-token
    # lookup of contract / invoice IDs that dense embeddings tend to miss.
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="text",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.MULTILINGUAL,
                min_token_len=2,
                max_token_len=64,
                lowercase=True,
            ),
        )
    except Exception as exc:
        # Older Qdrant versions or unsupported tokenizer settings — non-fatal.
        logger.debug("Text payload index create skipped: %s", exc)


def store_chunks(chunks: list[DocumentChunk]) -> int:
    """Embed and store document chunks in Qdrant. Returns count stored."""
    if not chunks:
        return 0

    settings = get_settings()
    client = get_qdrant_client()
    encoder = get_encoder()

    vector_size = encoder.get_sentence_embedding_dimension()
    ensure_collection(client, settings.qdrant_collection, vector_size)

    texts = [c.normalized_text for c in chunks]
    embeddings = encoder.encode(texts, batch_size=32, show_progress_bar=False)

    points = []
    for chunk, embedding in zip(chunks, embeddings):
        # Deterministic point ID — re-uploading the same doc upserts in place
        # instead of creating duplicate vector points.
        point_id = canonical_chunk_point_id(chunk.doc_id, chunk.chunk_index)
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "normalized_text": chunk.normalized_text,
                    "page_num": chunk.page_num,
                    "chunk_index": chunk.chunk_index,
                    "language": chunk.language,
                    "entities": chunk.entities,
                },
            )
        )

    batch_size = 32
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points[i : i + batch_size],
        )
    logger.info("Stored %d chunks in Qdrant for collection %s", len(points), settings.qdrant_collection)
    return len(points)


def semantic_search(
    query: str,
    doc_ids: Optional[list[str]] = None,
    top_k: int = 10,
    language_filter: Optional[str] = None,
) -> list[dict]:
    """Semantic search over chunks. Optionally filter by doc_ids or language."""
    settings = get_settings()
    client = get_qdrant_client()
    encoder = get_encoder()

    query_vector = encoder.encode(query).tolist()

    must_conditions = []
    if doc_ids:
        # Match any of the provided doc_ids
        must_conditions.append(
            Filter(
                should=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                    for doc_id in doc_ids
                ]
            )
        )
    if language_filter:
        must_conditions.append(
            FieldCondition(key="language", match=MatchValue(value=language_filter))
        )

    search_filter = Filter(must=must_conditions) if must_conditions else None

    # Qdrant client API differs by version:
    # - older: client.search(...)
    # - newer (1.17+): client.query_points(...)
    if hasattr(client, "search"):
        results = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        )
    else:
        query_response = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        )
        results = query_response.points

    return [
        {
            "score": hit.score,
            "doc_id": hit.payload.get("doc_id"),
            "text": hit.payload.get("text"),
            "page_num": hit.payload.get("page_num"),
            "language": hit.payload.get("language"),
            "chunk_id": hit.payload.get("chunk_id"),
        }
        for hit in results
    ]


def get_all_chunks_for_docs(doc_ids: list[str], batch_size: int = 100) -> list[dict]:
    """Retrieve ALL stored chunks for the given doc_ids (no embedding query, scroll-based)."""
    if not doc_ids:
        return []
    settings = get_settings()
    client = get_qdrant_client()
    search_filter = Filter(
        should=[
            FieldCondition(key="doc_id", match=MatchValue(value=did))
            for did in doc_ids
        ]
    )
    all_results: list[dict] = []
    offset = None
    while True:
        scroll_kwargs: dict = dict(
            collection_name=settings.qdrant_collection,
            scroll_filter=search_filter,
            limit=batch_size,
            with_payload=True,
        )
        if offset is not None:
            scroll_kwargs["offset"] = offset
        records, next_offset = client.scroll(**scroll_kwargs)
        for rec in records:
            all_results.append({
                "source": "vector_scroll",
                "doc_id": rec.payload.get("doc_id", ""),
                "page": rec.payload.get("page_num", 0),
                "language": rec.payload.get("language", ""),
                "score": 1.0,
                "text": rec.payload.get("text", ""),
            })
        if next_offset is None or not records:
            break
        offset = next_offset
    return all_results


def keyword_search(
    keyword: str,
    doc_ids: Optional[list[str]] = None,
    top_k: int = 10,
) -> list[dict]:
    """Full-text keyword search over chunk `text` payload.

    Complements :func:`semantic_search` for exact-token lookup of identifiers
    (e.g. ``"INV-2024-0837"``, ``"C-MOH-2023-041"``) where dense embeddings
    are unreliable. Requires the TEXT payload index created by
    :func:`ensure_collection`.
    """
    kw = (keyword or "").strip()
    if not kw:
        return []

    settings = get_settings()
    client = get_qdrant_client()

    must = [FieldCondition(key="text", match=MatchText(text=kw))]
    if doc_ids:
        must.append(
            Filter(
                should=[
                    FieldCondition(key="doc_id", match=MatchValue(value=did))
                    for did in doc_ids
                ]
            )
        )

    try:
        records, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=Filter(must=must),
            limit=top_k,
            with_payload=True,
        )
    except Exception as exc:
        logger.debug("keyword_search failed (text index missing?): %s", exc)
        return []

    return [
        {
            "score": 1.0,
            "doc_id": rec.payload.get("doc_id"),
            "text": rec.payload.get("text"),
            "page_num": rec.payload.get("page_num"),
            "language": rec.payload.get("language"),
            "chunk_id": rec.payload.get("chunk_id"),
            "source": "keyword",
        }
        for rec in records
    ]


def delete_doc_chunks(doc_id: str) -> None:
    """Remove all chunks belonging to a document."""
    settings = get_settings()
    client = get_qdrant_client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )


def delete_docs_chunks(doc_ids: list[str]) -> None:
    """Remove all chunks belonging to any of the given documents in one request."""
    if not doc_ids:
        return
    settings = get_settings()
    client = get_qdrant_client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            should=[FieldCondition(key="doc_id", match=MatchValue(value=did)) for did in doc_ids]
        ),
    )
    logger.info("Deleted Qdrant chunks for %d doc(s)", len(doc_ids))
