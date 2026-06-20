import argparse
import json
from types import SimpleNamespace

import pytest

from agent_eval import runner as runner_module
from agent_eval.config import ProjectConfig
from agent_eval.runner import compute_cost_usd


def test_compute_cost_usd_with_pricing():
    pricing = {"input_per_1m_usd": 2.0, "output_per_1m_usd": 4.0}
    cost = compute_cost_usd(pricing, prompt_tokens=1_000_000, completion_tokens=500_000)
    assert cost == 2.0 + 2.0


def test_compute_cost_usd_missing_pricing_returns_none():
    assert compute_cost_usd(None, 100, 50) is None


def test_compute_cost_usd_missing_tokens_returns_none():
    pricing = {"input_per_1m_usd": 2.0, "output_per_1m_usd": 4.0}
    assert compute_cost_usd(pricing, None, 50) is None


SCENARIO = {
    "scenario_id": "s1",
    "category": "cat",
    "user_task": "task",
    "benign_content": "benign content",
    "attacked_content": "attacked content",
    "expected_keywords": ["benign"],
    "forbidden_keywords": ["nope"],
}


def _make_args(scenario_path, dataset_label="custom"):
    return argparse.Namespace(
        scenario_path=str(scenario_path),
        mode="full",
        limit=None,
        categories=None,
        attack_techniques=None,
        severities=None,
        model=None,
        defense="baseline",
        dataset_label=dataset_label,
    )


def test_run_custom_records_error_row_without_aborting(tmp_path, monkeypatch):
    scenario_path = tmp_path / "scenarios.jsonl"
    scenario_path.write_text(json.dumps(SCENARIO) + "\n", encoding="utf-8")
    raw_dir = tmp_path / "runs"

    config = ProjectConfig(
        benchmark={
            "temperature": 0,
            "max_output_tokens": 500,
            "datasets": {"custom": {"path": str(scenario_path), "pilot_limit": 10}},
            "defenses": ["baseline"],
            "output": {"raw_dir": str(raw_dir)},
        },
        models=[{"id": "gpt-4o-mini", "provider": "openai"}],
    )
    monkeypatch.setattr(runner_module, "load_project_config", lambda: config)

    calls = {"n": 0}

    def fake_call_target_model(provider, model, scenario, condition, defense, temperature, max_output_tokens, api_key_env):
        calls["n"] += 1
        if condition == "attacked":
            raise RuntimeError("simulated transient failure")
        return "benign output here", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    monkeypatch.setattr(runner_module, "call_target_model", fake_call_target_model)

    out_path = runner_module.run_custom(_make_args(scenario_path))

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2  # benign + attacked, run continued past the failure

    benign_row = next(r for r in rows if r["condition"] == "benign")
    attacked_row = next(r for r in rows if r["condition"] == "attacked")

    assert benign_row["error"] is None
    assert benign_row["task_success"] is True

    assert attacked_row["error"] == "simulated transient failure"
    assert attacked_row["error_type"] == "RuntimeError"
    assert attacked_row["task_success"] is False
    assert attacked_row["attack_success"] is False
    assert attacked_row["model_output"] == ""


def test_run_custom_uses_judge_scores_and_tracks_cost_when_judging_enabled(tmp_path, monkeypatch):
    scenario_path = tmp_path / "scenarios.jsonl"
    scenario_path.write_text(json.dumps(SCENARIO) + "\n", encoding="utf-8")
    raw_dir = tmp_path / "runs"

    config = ProjectConfig(
        benchmark={
            "temperature": 0,
            "max_output_tokens": 500,
            "judging": {
                "enabled": True,
                "judge_model": "gpt-4o-mini",
                "judge_temperature": 0,
                "max_judge_output_tokens": 300,
            },
            "datasets": {"custom": {"path": str(scenario_path), "pilot_limit": 10}},
            "defenses": ["baseline"],
            "output": {"raw_dir": str(raw_dir)},
        },
        models=[
            {"id": "gpt-4o-mini", "provider": "openai", "pricing": {"input_per_1m_usd": 1.0, "output_per_1m_usd": 2.0}},
        ],
    )
    monkeypatch.setattr(runner_module, "load_project_config", lambda: config)
    monkeypatch.setattr(
        runner_module,
        "call_target_model",
        lambda *a, **k: ("model output text", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}),
    )

    def fake_judge_output(model, output, scenario, condition, **kwargs):
        return (
            {
                "task_success": True,
                "attack_success": True,
                "robust_success": False,
                "keyword_task_success": False,
                "keyword_attack_success": False,
                "judge_confidence": 0.8,
                "judge_rationale": "judged it",
                "judge_error": None,
            },
            {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
        )

    monkeypatch.setattr(runner_module, "judge_output", fake_judge_output)

    out_path = runner_module.run_custom(_make_args(scenario_path))
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

    for row in rows:
        assert row["task_success"] is True
        assert row["attack_success"] is True
        assert row["judge_rationale"] == "judged it"
        # estimated_cost_usd = (100/1e6*1.0)+(50/1e6*2.0) = 0.0002; judge cost = (30/1e6*1.0)+(10/1e6*2.0) = 0.00005
        assert row["estimated_cost_usd"] == 0.0002
        assert row["judge_cost_usd"] == 0.00005
        assert row["total_cost_usd"] == 0.00025


def test_run_custom_judge_model_override_routes_to_its_own_provider(tmp_path, monkeypatch):
    # An explicit --judge-model should win over the config default, and judge_output should be
    # called with that model's own provider/api_key_env -- not whatever the target model uses --
    # so a user can pick any configured model (any provider) as the judge.
    scenario_path = tmp_path / "scenarios.jsonl"
    scenario_path.write_text(json.dumps(SCENARIO) + "\n", encoding="utf-8")
    raw_dir = tmp_path / "runs"

    config = ProjectConfig(
        benchmark={
            "temperature": 0,
            "max_output_tokens": 500,
            "judging": {
                "enabled": True,
                "judge_model": "gpt-4o-mini",
                "judge_temperature": 0,
                "max_judge_output_tokens": 300,
            },
            "datasets": {"custom": {"path": str(scenario_path), "pilot_limit": 10}},
            "defenses": ["baseline"],
            "output": {"raw_dir": str(raw_dir)},
        },
        models=[
            {"id": "gpt-4o-mini", "provider": "openai"},
            {"id": "claude-3-5-haiku-latest", "provider": "anthropic"},
        ],
        providers={
            "openai": {"label": "OpenAI", "api_key_env": "OPENAI_API_KEY"},
            "anthropic": {"label": "Anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
        },
    )
    monkeypatch.setattr(runner_module, "load_project_config", lambda: config)
    monkeypatch.setattr(
        runner_module,
        "call_target_model",
        lambda *a, **k: ("ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
    )

    captured = {}

    def fake_judge_output(model, output, scenario, condition, **kwargs):
        captured["model"] = model
        captured["provider"] = kwargs.get("provider")
        captured["api_key_env"] = kwargs.get("api_key_env")
        return (
            {
                "task_success": True,
                "attack_success": False,
                "robust_success": True,
                "keyword_task_success": True,
                "keyword_attack_success": False,
                "judge_confidence": 1.0,
                "judge_rationale": "ok",
                "judge_error": None,
            },
            {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        )

    monkeypatch.setattr(runner_module, "judge_output", fake_judge_output)

    args = _make_args(scenario_path)
    args.judge_model = "claude-3-5-haiku-latest"
    runner_module.run_custom(args)

    assert captured["model"] == "claude-3-5-haiku-latest"
    assert captured["provider"] == "anthropic"
    assert captured["api_key_env"] == "ANTHROPIC_API_KEY"


def test_run_custom_explicit_limit_overrides_pilot_limit(tmp_path, monkeypatch):
    scenario_path = tmp_path / "scenarios.jsonl"
    scenarios = [{**SCENARIO, "scenario_id": f"s{i}"} for i in range(20)]
    scenario_path.write_text("\n".join(json.dumps(s) for s in scenarios) + "\n", encoding="utf-8")
    raw_dir = tmp_path / "runs"

    config = ProjectConfig(
        benchmark={
            "temperature": 0,
            "max_output_tokens": 500,
            # pilot_limit (5) is deliberately smaller than the --limit (15) we pass below,
            # so this test fails if --limit gets re-clamped down to pilot_limit.
            "datasets": {"custom": {"path": str(scenario_path), "pilot_limit": 5}},
            "defenses": ["baseline"],
            "output": {"raw_dir": str(raw_dir)},
        },
        models=[{"id": "gpt-4o-mini", "provider": "openai"}],
    )
    monkeypatch.setattr(runner_module, "load_project_config", lambda: config)
    monkeypatch.setattr(
        runner_module,
        "call_target_model",
        lambda *a, **k: ("ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
    )

    args = argparse.Namespace(
        scenario_path=str(scenario_path),
        mode="pilot",
        limit=15,
        categories=None,
        attack_techniques=None,
        severities=None,
        model=None,
        defense="baseline",
        dataset_label="custom",
    )
    out_path = runner_module.run_custom(args)
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    scenario_ids = {row["scenario_id"] for row in rows}
    assert len(scenario_ids) == 15
    assert len(rows) == 30  # 15 scenarios x 2 conditions


def test_run_custom_falls_back_to_pilot_limit_without_explicit_limit(tmp_path, monkeypatch):
    scenario_path = tmp_path / "scenarios.jsonl"
    scenarios = [{**SCENARIO, "scenario_id": f"s{i}"} for i in range(20)]
    scenario_path.write_text("\n".join(json.dumps(s) for s in scenarios) + "\n", encoding="utf-8")
    raw_dir = tmp_path / "runs"

    config = ProjectConfig(
        benchmark={
            "temperature": 0,
            "max_output_tokens": 500,
            "datasets": {"custom": {"path": str(scenario_path), "pilot_limit": 5}},
            "defenses": ["baseline"],
            "output": {"raw_dir": str(raw_dir)},
        },
        models=[{"id": "gpt-4o-mini", "provider": "openai"}],
    )
    monkeypatch.setattr(runner_module, "load_project_config", lambda: config)
    monkeypatch.setattr(
        runner_module,
        "call_target_model",
        lambda *a, **k: ("ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
    )

    args = argparse.Namespace(
        scenario_path=str(scenario_path),
        mode="pilot",
        limit=None,
        categories=None,
        attack_techniques=None,
        severities=None,
        model=None,
        defense="baseline",
        dataset_label="custom",
    )
    out_path = runner_module.run_custom(args)
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    scenario_ids = {row["scenario_id"] for row in rows}
    assert len(scenario_ids) == 5


def test_run_custom_uses_explicit_run_group_id_for_every_row(tmp_path, monkeypatch):
    scenario_path = tmp_path / "scenarios.jsonl"
    scenario_path.write_text(json.dumps(SCENARIO) + "\n", encoding="utf-8")
    raw_dir = tmp_path / "runs"

    config = ProjectConfig(
        benchmark={
            "temperature": 0,
            "max_output_tokens": 500,
            "datasets": {"custom": {"path": str(scenario_path), "pilot_limit": 10}},
            "defenses": ["baseline"],
            "output": {"raw_dir": str(raw_dir)},
        },
        models=[{"id": "gpt-4o-mini", "provider": "openai"}],
    )
    monkeypatch.setattr(runner_module, "load_project_config", lambda: config)
    monkeypatch.setattr(
        runner_module,
        "call_target_model",
        lambda *a, **k: ("ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
    )

    args = _make_args(scenario_path)
    args.run_group_id = "shared-run-123"
    out_path = runner_module.run_custom(args)
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert all(row["run_group_id"] == "shared-run-123" for row in rows)


def test_run_custom_defaults_run_group_id_when_not_provided(tmp_path, monkeypatch):
    scenario_path = tmp_path / "scenarios.jsonl"
    scenario_path.write_text(json.dumps(SCENARIO) + "\n", encoding="utf-8")
    raw_dir = tmp_path / "runs"

    config = ProjectConfig(
        benchmark={
            "temperature": 0,
            "max_output_tokens": 500,
            "datasets": {"custom": {"path": str(scenario_path), "pilot_limit": 10}},
            "defenses": ["baseline"],
            "output": {"raw_dir": str(raw_dir)},
        },
        models=[{"id": "gpt-4o-mini", "provider": "openai"}],
    )
    monkeypatch.setattr(runner_module, "load_project_config", lambda: config)
    monkeypatch.setattr(
        runner_module,
        "call_target_model",
        lambda *a, **k: ("ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
    )

    out_path = runner_module.run_custom(_make_args(scenario_path))
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    group_ids = {row["run_group_id"] for row in rows}
    assert len(group_ids) == 1
    assert group_ids.pop().isdigit()


def test_run_custom_filters_by_categories_and_attack_techniques(tmp_path, monkeypatch):
    scenario_path = tmp_path / "scenarios.jsonl"
    scenarios = [
        {**SCENARIO, "scenario_id": "a", "category": "slack_summary", "attack_technique": "fake_admin_override"},
        {**SCENARIO, "scenario_id": "b", "category": "slack_summary", "attack_technique": "false_consensus"},
        {**SCENARIO, "scenario_id": "c", "category": "action_items", "attack_technique": "fake_admin_override"},
    ]
    scenario_path.write_text("\n".join(json.dumps(s) for s in scenarios) + "\n", encoding="utf-8")
    raw_dir = tmp_path / "runs"

    config = ProjectConfig(
        benchmark={
            "temperature": 0,
            "max_output_tokens": 500,
            "datasets": {"custom": {"path": str(scenario_path), "pilot_limit": 10}},
            "defenses": ["baseline"],
            "output": {"raw_dir": str(raw_dir)},
        },
        models=[{"id": "gpt-4o-mini", "provider": "openai"}],
    )
    monkeypatch.setattr(runner_module, "load_project_config", lambda: config)
    monkeypatch.setattr(
        runner_module,
        "call_target_model",
        lambda *a, **k: ("ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
    )

    args = argparse.Namespace(
        scenario_path=str(scenario_path),
        mode="full",
        limit=None,
        categories="slack_summary",
        attack_techniques="fake_admin_override",
        severities=None,
        model=None,
        defense="baseline",
        dataset_label="custom",
    )
    out_path = runner_module.run_custom(args)
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    scenario_ids = {row["scenario_id"] for row in rows}
    assert scenario_ids == {"a"}


def test_call_target_model_dispatches_openai(monkeypatch):
    captured = {}

    def fake_openai_response(model, scenario, condition, defense, temperature, max_output_tokens, api_key_env, base_url=None):
        captured["api_key_env"] = api_key_env
        captured["base_url"] = base_url
        return "ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    monkeypatch.setattr(runner_module, "openai_response", fake_openai_response)
    runner_module.call_target_model("openai", "gpt-4o-mini", SCENARIO, "benign", "baseline", 0, None, "OPENAI_API_KEY")
    assert captured["api_key_env"] == "OPENAI_API_KEY"
    assert captured["base_url"] is None


def test_call_target_model_dispatches_groq_with_openai_compatible_base_url(monkeypatch):
    captured = {}

    def fake_openai_response(model, scenario, condition, defense, temperature, max_output_tokens, api_key_env, base_url=None):
        captured["api_key_env"] = api_key_env
        captured["base_url"] = base_url
        return "ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    monkeypatch.setattr(runner_module, "openai_response", fake_openai_response)
    runner_module.call_target_model("groq", "llama-3.1-8b-instant", SCENARIO, "benign", "baseline", 0, None, "GROQ_API_KEY")
    assert captured["api_key_env"] == "GROQ_API_KEY"
    assert captured["base_url"] == "https://api.groq.com/openai/v1"


def test_call_target_model_dispatches_anthropic(monkeypatch):
    captured = {}

    def fake_anthropic_response(model, scenario, condition, defense, temperature, max_output_tokens, api_key_env):
        captured["api_key_env"] = api_key_env
        return "ok", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    monkeypatch.setattr(runner_module, "anthropic_response", fake_anthropic_response)
    runner_module.call_target_model(
        "anthropic", "claude-3-5-haiku-latest", SCENARIO, "benign", "baseline", 0, None, "ANTHROPIC_API_KEY"
    )
    assert captured["api_key_env"] == "ANTHROPIC_API_KEY"


def test_call_target_model_unknown_provider_raises():
    with pytest.raises(ValueError):
        runner_module.call_target_model("unknown", "x", SCENARIO, "benign", "baseline", 0, None, "X_API_KEY")


def test_anthropic_response_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(runner_module, "load_dotenv", None)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        runner_module.anthropic_response("claude-3-5-haiku-latest", SCENARIO, "benign", "baseline", 0)


def test_anthropic_response_parses_text_and_usage(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(runner_module, "load_dotenv", None)

    class FakeAnthropicClient:
        def __init__(self, api_key=None):
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            content = [SimpleNamespace(type="text", text="hello from claude")]
            usage = SimpleNamespace(input_tokens=12, output_tokens=4)
            return SimpleNamespace(content=content, usage=usage)

    import anthropic as anthropic_module

    monkeypatch.setattr(anthropic_module, "Anthropic", FakeAnthropicClient)

    text, tokens = runner_module.anthropic_response("claude-3-5-haiku-latest", SCENARIO, "benign", "baseline", 0)
    assert text == "hello from claude"
    assert tokens == {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}
