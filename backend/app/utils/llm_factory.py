from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from app.config import get_settings


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Return a ChatOpenAI instance pointed at OpenRouter.

    When OPENROUTER_REASONING=true, the reasoning parameter is sent via
    extra_body (not model_kwargs) so OpenRouter receives it correctly in
    the HTTP request body without it being treated as an OpenAI API param.
    """
    settings = get_settings()

    extra_body: dict = {}
    if settings.openrouter_reasoning:
        extra_body["reasoning"] = {"enabled": True}

    return ChatOpenAI(
        model=settings.openrouter_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=temperature,
        extra_body=extra_body if extra_body else None,
        default_headers={
            "HTTP-Referer": "https://github.com/auditmind",
            "X-Title": "AuditMind",
        },
    )
