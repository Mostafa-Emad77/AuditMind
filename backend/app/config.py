from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # LLM — OpenRouter (OpenAI-compatible)
    openrouter_api_key: str = Field(default="")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_model: str = Field(
        default="tencent/hy4-preview",
        description="Any model slug from https://openrouter.ai/models",
    )
    openrouter_model_ner_arabic: str = Field(
        default="google/gemini-2.5-flash-lite",
        description="NER / entity extraction only; never substituted by OPENROUTER_MODEL",
    )
    openrouter_model_relation: str = Field(
        default="minimax/minimax-m2.5",
        description="Cross-checker / contradiction logic only; never substituted by OPENROUTER_MODEL",
    )
    openrouter_reasoning: bool = Field(
        default=True,
        description="Reasoning for the general/planning model. Sent explicitly either way",
    )
    openrouter_reasoning_relation: bool = Field(
        default=True,
        description=(
            "Reasoning for the cross-checker model. On by default: adjudication is the "
            "precision-critical stage, and some models (minimax-m2.5) reject a request "
            "that tries to disable it."
        ),
    )

    @field_validator("openrouter_model_ner_arabic", "openrouter_model_relation", mode="before")
    @classmethod
    def _stage_model_slugs_non_empty(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            raise ValueError("OpenRouter stage model slug must be non-empty")
        return v

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str = Field(default="")
    qdrant_collection: str = Field(default="auditmind_docs")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")
    neo4j_database: str = Field(
        default="neo4j",
        description="Neo4j database name (Aura default is usually 'neo4j')",
    )

    # Embeddings (OpenAI-compatible /embeddings via OpenRouter, uses OPENROUTER_API_KEY)
    embedding_model: str = Field(default="openai/text-embedding-3-small")
    embedding_dimension: int = Field(
        default=1536,
        description="Vector size of the embedding model.",
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis Cloud URL — redis://:[password]@[host]:[port]",
    )

    # Extraction speed — concurrency and pacing
    extraction_concurrency: int = Field(
        default=3,
        description="Max parallel LLM NER calls during extraction (paid tier: 3+; free tier: 1)",
    )
    extraction_chunk_sleep: float = Field(
        default=0.0,
        description="Seconds to sleep between extraction batches (0 for paid tier; 4.0 for free tier)",
    )

    # Extraction recall — adaptive retrieval
    extraction_top_k_base: int = Field(
        default=50,
        description="Base top_k for extraction queries on small docs (<=5 pages)",
    )
    extraction_top_k_max: int = Field(
        default=80,
        description="Hard cap on merged chunks passed to entity extraction per document",
    )
    extraction_chunk_cap: Optional[int] = Field(
        default=None,
        description="Extra cap inside the extractor; None = rely on extraction_top_k_max",
    )

    # Cross-checker precision gates
    finding_min_evidence_for_critical: int = Field(
        default=2,
        description="Minimum distinct evidence snippets required to emit a critical finding from LLM adjudication",
    )
    finding_min_confidence_llm: float = Field(
        default=0.65,
        description="Minimum confidence score for LLM-only adjudicated findings to be accepted",
    )
    finding_min_confidence_graph: float = Field(
        default=0.55,
        description="Minimum confidence score for graph-sourced findings to be accepted",
    )

    # Upload limits
    max_upload_file_size_mb: int = Field(
        default=20, description="Max size (MB) for a single uploaded PDF"
    )
    max_upload_files: int = Field(
        default=10, description="Max number of files accepted in a single upload request"
    )
    max_upload_pages: int = Field(
        default=50, description="Max page count for a single uploaded PDF"
    )

    # LLM reliability
    llm_timeout_seconds: float = Field(
        default=60.0, description="Per-request timeout for LLM calls"
    )
    llm_max_retries: int = Field(
        default=2, description="Max automatic retries on transient LLM call failures"
    )
    llm_max_output_tokens: int = Field(
        default=2048,
        description="Cap on completion tokens per call (all prompts ask for compact JSON)",
    )
    extraction_llm_table_repair: bool = Field(
        default=True,
        description=(
            "Let an LLM rebuild rows of a flattened table that no deterministic layer "
            "could recover. Output is accepted only if it conserves every number."
        ),
    )
    retriever_llm_query_entities: bool = Field(
        default=False,
        description="LLM-extract graph entities per checklist query (off: regex suffices)",
    )

    # Auth
    api_keys: str = Field(
        default="",
        description="Comma-separated list of valid API keys for /api/* routes. Empty = auth disabled.",
    )

    @property
    def api_keys_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    # App
    cors_origins: str = Field(default="http://localhost:3000")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
