from agent_eval.config import (
    ProjectConfig,
    get_api_key_env,
    get_model_provider,
    has_provider_api_key,
    missing_api_key_models,
)


def _config():
    return ProjectConfig(
        benchmark={},
        models=[
            {"id": "gpt-4o-mini", "provider": "openai"},
            {"id": "claude-3-5-haiku-latest", "provider": "anthropic"},
        ],
        providers={
            "openai": {"label": "OpenAI", "api_key_env": "OPENAI_API_KEY"},
            "anthropic": {"label": "Anthropic (Claude)", "api_key_env": "ANTHROPIC_API_KEY"},
        },
    )


def test_get_model_provider():
    config = _config()
    assert get_model_provider(config, "gpt-4o-mini") == "openai"
    assert get_model_provider(config, "claude-3-5-haiku-latest") == "anthropic"
    assert get_model_provider(config, "unknown-model") is None


def test_get_api_key_env():
    config = _config()
    assert get_api_key_env(config, "openai") == "OPENAI_API_KEY"
    assert get_api_key_env(config, "nonexistent") is None


def test_has_provider_api_key(monkeypatch):
    config = _config()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert has_provider_api_key(config, "openai") is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert has_provider_api_key(config, "openai") is True


def test_missing_api_key_models(monkeypatch):
    config = _config()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    missing = missing_api_key_models(config, ["gpt-4o-mini", "claude-3-5-haiku-latest"])
    assert missing == [("claude-3-5-haiku-latest", "anthropic", "ANTHROPIC_API_KEY")]


def test_missing_api_key_models_empty_when_all_keys_present(monkeypatch):
    config = _config()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    assert missing_api_key_models(config, ["gpt-4o-mini", "claude-3-5-haiku-latest"]) == []
