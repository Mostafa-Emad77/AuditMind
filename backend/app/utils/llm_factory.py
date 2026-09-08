from functools import lru_cache
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from app.config import get_settings

ModelRole = Literal["ner_arabic", "relation"]


@lru_cache(maxsize=None)
def get_llm(
    temperature: float = 0.0,
    role: ModelRole | None = None,
) -> BaseChatModel:
    """Configured chat model (OpenRouter, OpenAI-compatible), cached per (temperature, role).

    role: ner_arabic → extraction model; relation → cross-checker model; None → general.
    """
    settings = get_settings()

    if role == "ner_arabic":
        model = settings.openrouter_model_ner_arabic
    elif role == "relation":
        model = settings.openrouter_model_relation
    else:
        model = settings.openrouter_model

    # Set reasoning explicitly both ways: some models think by default and burn
    # thousands of tokens on tiny prompts when the flag is merely omitted, while
    # others (minimax-m2.5) reject a request that tries to disable it outright.
    if role == "relation":
        reasoning_on = settings.openrouter_reasoning_relation
    elif role is None:
        reasoning_on = settings.openrouter_reasoning
    else:
        reasoning_on = False  # NER is mechanical extraction; thinking is wasted there
    extra_body: dict = {"reasoning": {"enabled": reasoning_on}}

    return ChatOpenAI(
        model=model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        max_tokens=settings.llm_max_output_tokens,
        extra_body=extra_body,
        default_headers={
            "HTTP-Referer": "https://github.com/auditmind",
            "X-Title": "AuditMind",
        },
    )
