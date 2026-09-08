"""Reasoning must be set explicitly, per role.

Two opposite failures made this necessary: omitting the flag let a model that thinks
by default burn thousands of tokens on a 100-token prompt, and sending
`enabled: false` made minimax-m2.5 reject the request outright ("Reasoning is
mandatory for this endpoint and cannot be disabled") — which would have failed every
adjudication call.
"""
import pytest

from app.config import get_settings
from app.utils.llm_factory import get_llm


@pytest.fixture(autouse=True)
def _fresh_clients():
    """get_llm is lru_cached on (temperature, role); clear it between cases."""
    get_llm.cache_clear()
    yield
    get_llm.cache_clear()


def _reasoning(role, temperature=0.0):
    return get_llm(temperature=temperature, role=role).extra_body["reasoning"]["enabled"]


class TestPerRoleReasoning:
    def test_relation_role_enables_reasoning_by_default(self):
        """The adjudication model must never be sent enabled:false — it 400s."""
        assert _reasoning("relation") is True

    def test_relation_role_is_configurable(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "openrouter_reasoning_relation", False)
        get_llm.cache_clear()
        assert _reasoning("relation") is False

    def test_ner_role_never_reasons(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "openrouter_reasoning", True)
        get_llm.cache_clear()
        assert _reasoning("ner_arabic") is False

    def test_general_role_follows_its_own_setting(self, monkeypatch):
        settings = get_settings()
        for value in (True, False):
            monkeypatch.setattr(settings, "openrouter_reasoning", value)
            get_llm.cache_clear()
            assert _reasoning(None) is value


class TestFlagIsAlwaysExplicit:
    @pytest.mark.parametrize("role", [None, "relation", "ner_arabic"])
    def test_every_role_sends_the_flag(self, role):
        """Omitting it is what let a reasoning-by-default model run away."""
        body = get_llm(temperature=0.0, role=role).extra_body
        assert "reasoning" in body
        assert isinstance(body["reasoning"]["enabled"], bool)

    def test_output_tokens_are_capped(self):
        settings = get_settings()
        assert get_llm(temperature=0.0, role="relation").max_tokens == settings.llm_max_output_tokens


class TestRoleSelectsModel:
    def test_each_role_uses_its_own_slug(self):
        settings = get_settings()
        assert get_llm(temperature=0.0, role="relation").model_name == settings.openrouter_model_relation
        get_llm.cache_clear()
        assert get_llm(temperature=0.0, role="ner_arabic").model_name == settings.openrouter_model_ner_arabic
        get_llm.cache_clear()
        assert get_llm(temperature=0.0, role=None).model_name == settings.openrouter_model
