from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectConfig:
    benchmark: dict[str, Any]
    models: list[dict[str, Any]]
    providers: dict[str, Any] = field(default_factory=dict)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def load_project_config(
    benchmark_path: str | Path = ROOT / "configs" / "benchmark.yaml",
    models_path: str | Path = ROOT / "configs" / "models.yaml",
) -> ProjectConfig:
    benchmark = load_yaml(benchmark_path)
    models_doc = load_yaml(models_path)
    models = models_doc.get("models", [])
    providers = models_doc.get("providers", {})
    return ProjectConfig(benchmark=benchmark, models=models, providers=providers)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def get_model_pricing(config: ProjectConfig, model_id: str) -> dict[str, float] | None:
    for model in config.models:
        if model.get("id") == model_id:
            return model.get("pricing")
    return None


def get_model_provider(config: ProjectConfig, model_id: str) -> str | None:
    for model in config.models:
        if model.get("id") == model_id:
            return model.get("provider")
    return None


def get_provider_info(config: ProjectConfig, provider_id: str) -> dict[str, Any] | None:
    return config.providers.get(provider_id)


def get_api_key_env(config: ProjectConfig, provider_id: str) -> str | None:
    info = get_provider_info(config, provider_id)
    return info.get("api_key_env") if info else None


def has_provider_api_key(config: ProjectConfig, provider_id: str) -> bool:
    env_var = get_api_key_env(config, provider_id)
    return bool(env_var and os.getenv(env_var))


def missing_api_key_models(config: ProjectConfig, model_ids: list[str]) -> list[tuple[str, str, str]]:
    """Returns (model_id, provider_id, api_key_env) for every requested model whose
    provider's API key is not currently set in the environment."""
    missing = []
    for model_id in model_ids:
        provider_id = get_model_provider(config, model_id)
        if provider_id is None:
            continue
        env_var = get_api_key_env(config, provider_id)
        if env_var and not os.getenv(env_var):
            missing.append((model_id, provider_id, env_var))
    return missing
