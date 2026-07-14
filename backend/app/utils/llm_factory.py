from functools import lru_cache
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from app.config import get_settings

OpenRouterRole = Literal["ner_arabic", "relation"]


def _get_google_llm(role: OpenRouterRole | None, temperature: float) -> BaseChatModel:
    """Return a ChatGoogleGenerativeAI instance for Google AI Studio."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()

    if role == "ner_arabic":
        model = settings.google_model_ner_arabic
    elif role == "relation":
        model = settings.google_model_relation
    else:
        model = settings.google_model

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


@lru_cache(maxsize=None)
def get_llm(
    temperature: float = 0.0,
    role: OpenRouterRole | None = None,
) -> BaseChatModel:
    """
    Return the configured LLM instance.

    Cached on (temperature, role): the underlying clients are stateless and settings
    are fixed for the process lifetime, so we build each distinct client once instead
    of re-instantiating it on every call.

    When LLM_PROVIDER=google, uses Google AI Studio (Gemini) via langchain-google-genai.
    When LLM_PROVIDER=openrouter (default), uses OpenRouter via ChatOpenAI.

    ``role`` selects a dedicated model slug:
    ``ner_arabic`` → NER / entity extraction model.
    ``relation``   → cross-checker / contradiction model.
    When ``role`` is None, uses the general / planning / report model.
    """
    settings = get_settings()

    if settings.llm_provider == "google":
        return _get_google_llm(role=role, temperature=temperature)

    if role == "ner_arabic":
        model = settings.openrouter_model_ner_arabic
    elif role == "relation":
        model = settings.openrouter_model_relation
    else:
        model = settings.openrouter_model

    extra_body: dict = {}
    if role is None and settings.openrouter_reasoning:
        extra_body["reasoning"] = {"enabled": True}

    return ChatOpenAI(
        model=model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        extra_body=extra_body if extra_body else None,
        default_headers={
            "HTTP-Referer": "https://github.com/auditmind",
            "X-Title": "AuditMind",
        },
    )
