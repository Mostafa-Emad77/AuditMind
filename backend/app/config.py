from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # LLM — OpenRouter
    openrouter_api_key: str = Field(default="")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_model: str = Field(
        default="google/gemini-3-flash-preview",
        description="Any model slug from https://openrouter.ai/models",
    )
    openrouter_reasoning: bool = Field(
        default=True,
        description="Pass reasoning:{enabled:true} — supported by minimax-m2.5 and other reasoning models",
    )

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

    # Embeddings
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis Cloud URL — redis://:[password]@[host]:[port]",
    )

    # App
    app_env: str = Field(default="development")
    cors_origins: str = Field(default="http://localhost:3000")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
