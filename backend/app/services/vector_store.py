"""Qdrant vector store operations for AuditMind."""
import logging
from typing import Optional

from qdrant_client import QdrantClient
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
from app.config import get_settings
from app.models.schemas import DocumentChunk
from app.utils.canonical_id import canonical_chunk_point_id

logger = logging.getLogger(__name__)


class _Embedder:
    """Provider-agnostic embedding interface: `encode(texts)` and `.dimension`."""

    dimension: int

    def encode(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - interface
        raise NotImplementedError

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


class _OpenRouterEmbedder(_Embedder):
    """OpenAI-compatible /embeddings via OpenRouter (e.g. openai/text-embedding-3-small)."""

    def __init__(self, model: str, dimension: int, api_key: str, base_url: str) -> None:
        from langchain_openai import OpenAIEmbeddings

        self.dimension = dimension
        self._client = OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            # Send raw strings; tiktoken doesn't know OpenRouter slugs.
            check_embedding_ctx_length=False,
            chunk_size=64,
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # API rejects empty strings.
        return self._client.embed_documents([t if t.strip() else " " for t in texts])


_encoder: Optional[_Embedder] = None
_client: Optional[QdrantClient] = None
_collection_ready: Optional[str] = None


def get_encoder() -> _Embedder:
    global _encoder
    if _encoder is None:
        settings = get_settings()
        _encoder = _OpenRouterEmbedder(
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
        logger.info(
            "Embeddings: model=%s dim=%d",
            settings.embedding_model, _encoder.dimension,
        )
    return _encoder


def get_qdrant_client() -> QdrantClient:
    """Process-wide singleton."""
    global _client
    if _client is None:
        settings = get_settings()
        kwargs: dict = {"url": settings.qdrant_url, "timeout": 120}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        _client = QdrantClient(**kwargs)
    return _client


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = 1536) -> None:
    """Create the collection + payload indexes once per process."""
    global _collection_ready
    if _collection_ready == collection_name:
        return

    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection: %s (dim=%d)", collection_name, vector_size)
    else:
        # Vector size mismatch (model changed): recreate if empty, else fail loudly.
        try:
            info = client.get_collection(collection_name)
            current = info.config.params.vectors.size  # type: ignore[union-attr]
            if current != vector_size:
                if (info.points_count or 0) == 0:
                    client.delete_collection(collection_name)
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    )
                    logger.warning(
                        "Recreated empty Qdrant collection %s: dim %d -> %d",
                        collection_name, current, vector_size,
                    )
                else:
                    raise RuntimeError(
                        f"Qdrant collection {collection_name!r} has vector size {current} but the "
                        f"configured embedding model produces {vector_size}. Set QDRANT_COLLECTION "
                        f"to a new name (or delete the old collection) and re-upload documents."
                    )
        except RuntimeError:
            raise
        except Exception as exc:
            logger.debug("Could not verify collection vector size: %s", exc)
    # Payload indexes are required for filtered search; idempotent.
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
    # Full-text index for exact-token keyword_search().
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
        # Unsupported on older Qdrant — non-fatal.
        logger.debug("Text payload index create skipped: %s", exc)

    _collection_ready = collection_name


def init_collection() -> None:
    """Create the configured collection at startup, like the Neo4j schema init."""
    settings = get_settings()
    ensure_collection(
        get_qdrant_client(),
        settings.qdrant_collection,
        get_encoder().dimension,
    )


def store_chunks(chunks: list[DocumentChunk]) -> int:
    """Embed and store document chunks in Qdrant. Returns count stored."""
    if not chunks:
        return 0

    settings = get_settings()
    client = get_qdrant_client()
    encoder = get_encoder()

    ensure_collection(client, settings.qdrant_collection, encoder.dimension)

    texts = [c.normalized_text for c in chunks]
    embeddings = encoder.encode(texts)

    points = []
    for chunk, embedding in zip(chunks, embeddings):
        # Deterministic id → re-uploads upsert in place.
        point_id = canonical_chunk_point_id(chunk.doc_id, chunk.chunk_index)
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
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

    query_vector = encoder.encode_one(query)

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

    # search() on older clients, query_points() on 1.17+.
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


def count_chunks_for_docs(doc_ids: list[str]) -> dict[str, int]:
    """Exact stored-chunk count per doc_id — the ingestion half of the coverage report."""
    if not doc_ids:
        return {}
    settings = get_settings()
    client = get_qdrant_client()
    counts: dict[str, int] = {}
    for did in doc_ids:
        try:
            counts[did] = client.count(
                collection_name=settings.qdrant_collection,
                count_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=did))]),
                exact=True,
            ).count
        except Exception as exc:
            logger.warning("Chunk count failed for %s: %s", did, exc)
            counts[did] = -1
    return counts


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
    """Exact-token search over chunk text (identifiers dense embeddings miss)."""
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
