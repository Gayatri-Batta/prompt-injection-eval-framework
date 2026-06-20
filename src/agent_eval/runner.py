from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from agent_eval.agentdojo_adapter import (
    DEFAULT_ATTACK,
    build_agentdojo_command,
    map_model_to_agentdojo,
    parse_agentdojo_logdir,
)
from agent_eval.config import get_api_key_env, get_model_pricing, get_model_provider, load_project_config, resolve_path
from agent_eval.custom_dataset import filter_scenario_rows, read_jsonl
from agent_eval.defenses import DEFENSES, build_prompt, get_system_prompt
from agent_eval.judge import judge_output
from agent_eval.scoring import score_custom_output

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_ANTHROPIC_MAX_TOKENS = 1024

# Groq exposes an OpenAI-compatible Chat Completions endpoint, so it can be reached with the
# openai SDK by pointing it at a different base_url -- no separate client library needed.
PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
}


def compute_cost_usd(
    pricing: dict[str, float] | None, prompt_tokens: int | None, completion_tokens: int | None
) -> float | None:
    if not pricing or prompt_tokens is None or completion_tokens is None:
        return None
    input_cost = (prompt_tokens / 1_000_000) * pricing["input_per_1m_usd"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output_per_1m_usd"]
    return input_cost + output_cost


def _require_api_key(api_key_env: str, provider_label: str) -> None:
    if load_dotenv is not None:
        load_dotenv(resolve_path(".env"))
    if not os.getenv(api_key_env):
        raise RuntimeError(
            f"{api_key_env} is missing. Add it to `.env` as {api_key_env}=your_key_here "
            f"or set it in your terminal before running {provider_label} experiments."
        )


def openai_response(
    model: str,
    scenario: dict,
    condition: str,
    defense: str,
    temperature: float,
    max_output_tokens: int | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str | None = None,
) -> tuple[str, dict]:
    """Calls any OpenAI-compatible Chat Completions endpoint (OpenAI itself, or a
    compatible provider like Groq reached via a different base_url/api key)."""
    _require_api_key(api_key_env, provider_label=base_url or "OpenAI")

    try:
        from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI SDK is not installed. Run `pip install -r requirements.txt` before running these experiments."
        ) from exc

    # APIConnectionError covers its subclass APITimeoutError. Deliberately excludes
    # APIStatusError subclasses like BadRequestError/AuthenticationError, which won't
    # succeed on retry.
    retryable_exceptions = (RateLimitError, APIConnectionError, InternalServerError)
    client = OpenAI(api_key=os.getenv(api_key_env), base_url=base_url)
    kwargs = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": get_system_prompt(defense)},
            {"role": "user", "content": build_prompt(scenario, condition, defense)},
        ],
    }
    if max_output_tokens:
        kwargs["max_tokens"] = max_output_tokens

    attempt = 0
    while True:
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except retryable_exceptions:
            attempt += 1
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))

    usage = response.usage
    tokens = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    return response.choices[0].message.content or "", tokens


def anthropic_response(
    model: str,
    scenario: dict,
    condition: str,
    defense: str,
    temperature: float,
    max_output_tokens: int | None = None,
    api_key_env: str = "ANTHROPIC_API_KEY",
) -> tuple[str, dict]:
    """Calls Anthropic's Messages API, which has a different request/response shape than
    OpenAI's Chat Completions (system prompt is a top-level field, not a message; max_tokens
    is required; usage fields are named input_tokens/output_tokens)."""
    _require_api_key(api_key_env, provider_label="Anthropic")

    try:
        from anthropic import (
            Anthropic,
            APIConnectionError,
            InternalServerError,
            RateLimitError,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Anthropic SDK is not installed. Run `pip install -r requirements.txt` before running Anthropic experiments."
        ) from exc

    retryable_exceptions = (RateLimitError, APIConnectionError, InternalServerError)
    client = Anthropic(api_key=os.getenv(api_key_env))
    kwargs = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_output_tokens or DEFAULT_ANTHROPIC_MAX_TOKENS,
        "system": get_system_prompt(defense),
        "messages": [{"role": "user", "content": build_prompt(scenario, condition, defense)}],
    }

    attempt = 0
    while True:
        try:
            response = client.messages.create(**kwargs)
            break
        except retryable_exceptions:
            attempt += 1
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))

    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    tokens = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": (input_tokens + output_tokens) if input_tokens is not None and output_tokens is not None else None,
    }
    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return text, tokens


def call_target_model(
    provider: str,
    model: str,
    scenario: dict,
    condition: str,
    defense: str,
    temperature: float,
    max_output_tokens: int | None,
    api_key_env: str,
) -> tuple[str, dict]:
    """Dispatches a target-model call to the right provider implementation."""
    if provider == "anthropic":
        return anthropic_response(model, scenario, condition, defense, temperature, max_output_tokens, api_key_env)
    if provider in PROVIDER_BASE_URLS:
        return openai_response(
            model, scenario, condition, defense, temperature, max_output_tokens, api_key_env, PROVIDER_BASE_URLS[provider]
        )
    if provider == "openai":
        return openai_response(model, scenario, condition, defense, temperature, max_output_tokens, api_key_env)
    raise ValueError(f"Unknown provider '{provider}'. Add it to runner.call_target_model.")


def run_custom(args: argparse.Namespace) -> Path:
    config = load_project_config()
    benchmark = config.benchmark
    scenario_path = resolve_path(args.scenario_path or benchmark["datasets"]["custom"]["path"])
    scenarios = read_jsonl(scenario_path)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else None
    attack_techniques = (
        [t.strip() for t in args.attack_techniques.split(",") if t.strip()] if args.attack_techniques else None
    )
    severities = [s.strip() for s in args.severities.split(",") if s.strip()] if args.severities else None
    scenarios = filter_scenario_rows(
        scenarios, categories=categories, attack_techniques=attack_techniques, severities=severities
    )
    if args.limit:
        # An explicit --limit always wins -- it's a deliberate request for exactly N
        # scenarios, not a secondary cap on top of the pilot-mode default.
        scenarios = scenarios[: args.limit]
    elif args.mode == "pilot":
        scenarios = scenarios[: int(benchmark["datasets"]["custom"]["pilot_limit"])]

    models = [m["id"] for m in config.models]
    if args.model:
        models = [args.model]
    defenses = benchmark["defenses"]
    if args.defense:
        defenses = [args.defense]

    raw_dir = resolve_path(benchmark["output"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"custom_openai_{int(time.time())}.jsonl"
    run_group_id = getattr(args, "run_group_id", None) or str(int(time.time()))

    max_output_tokens = benchmark.get("max_output_tokens")
    judging_cfg = benchmark.get("judging") or {}
    judging_enabled = bool(judging_cfg.get("enabled", False))
    # An explicit --judge-model always wins over the config default, the same way --model
    # overrides the target-model list -- lets the dashboard offer a judge picker instead of
    # hardcoding one judge for every run.
    judge_model = getattr(args, "judge_model", None) or judging_cfg.get("judge_model")
    if judge_model:
        judge_provider = get_model_provider(config, judge_model) or "openai"
        judge_api_key_env = get_api_key_env(config, judge_provider) or "OPENAI_API_KEY"
        judge_base_url = PROVIDER_BASE_URLS.get(judge_provider)
    else:
        judge_provider = judge_api_key_env = judge_base_url = None
    judge_temperature = float(judging_cfg.get("judge_temperature", 0))
    max_judge_output_tokens = judging_cfg.get("max_judge_output_tokens")
    error_rows = 0
    success_rows = 0
    empty_judge_tokens = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    with out_path.open("w", encoding="utf-8") as handle:
        for model in models:
            provider = get_model_provider(config, model) or "openai"
            api_key_env = get_api_key_env(config, provider) or "OPENAI_API_KEY"
            for defense in defenses:
                for scenario in scenarios:
                    for condition in ["benign", "attacked"]:
                        start = time.perf_counter()
                        error: str | None = None
                        error_type: str | None = None
                        token_data = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
                        judge_token_data = dict(empty_judge_tokens)
                        try:
                            output, token_data = call_target_model(
                                provider,
                                model,
                                scenario,
                                condition,
                                defense,
                                float(benchmark.get("temperature", 0)),
                                max_output_tokens,
                                api_key_env,
                            )
                            if judging_enabled and judge_model:
                                scores, judge_token_data = judge_output(
                                    judge_model,
                                    output,
                                    scenario,
                                    condition,
                                    temperature=judge_temperature,
                                    max_output_tokens=max_judge_output_tokens,
                                    provider=judge_provider,
                                    api_key_env=judge_api_key_env,
                                    base_url=judge_base_url,
                                )
                            else:
                                keyword_scores = score_custom_output(output, scenario)
                                scores = {
                                    **keyword_scores,
                                    "keyword_task_success": keyword_scores["task_success"],
                                    "keyword_attack_success": keyword_scores["attack_success"],
                                    "judge_confidence": None,
                                    "judge_rationale": None,
                                    "judge_error": None,
                                }
                            success_rows += 1
                        except Exception as exc:  # noqa: BLE001 - record and continue, never abort the run
                            output = ""
                            scores = {
                                "task_success": False,
                                "attack_success": False,
                                "robust_success": False,
                                "keyword_task_success": False,
                                "keyword_attack_success": False,
                                "judge_confidence": None,
                                "judge_rationale": None,
                                "judge_error": None,
                            }
                            error = str(exc)[:2000]
                            error_type = type(exc).__name__
                            error_rows += 1
                        latency = time.perf_counter() - start
                        pricing = get_model_pricing(config, model)
                        estimated_cost_usd = compute_cost_usd(
                            pricing, token_data["prompt_tokens"], token_data["completion_tokens"]
                        )
                        if judging_enabled and judge_model:
                            judge_pricing = get_model_pricing(config, judge_model)
                            judge_cost_usd = compute_cost_usd(
                                judge_pricing, judge_token_data["prompt_tokens"], judge_token_data["completion_tokens"]
                            )
                            total_cost_usd = (
                                None
                                if estimated_cost_usd is None or judge_cost_usd is None
                                else estimated_cost_usd + judge_cost_usd
                            )
                        else:
                            judge_cost_usd = None
                            total_cost_usd = estimated_cost_usd
                        row = {
                            "run_id": str(uuid.uuid4()),
                            "run_group_id": run_group_id,
                            "model": model,
                            "dataset": args.dataset_label,
                            "scenario_id": scenario["scenario_id"],
                            "category": scenario["category"],
                            "attack_technique": scenario.get("attack_technique"),
                            "injection_position": scenario.get("injection_position"),
                            "severity": scenario.get("severity"),
                            "defense": defense,
                            "condition": condition,
                            "latency_seconds": latency,
                            "estimated_cost_usd": estimated_cost_usd,
                            "judge_cost_usd": judge_cost_usd,
                            "total_cost_usd": total_cost_usd,
                            "user_task": scenario["user_task"],
                            "untrusted_content": scenario["attacked_content"]
                            if condition == "attacked"
                            else scenario["benign_content"],
                            "model_output": output,
                            "error": error,
                            "error_type": error_type,
                            **token_data,
                            **scores,
                        }
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Completed run: {success_rows} succeeded, {error_rows} errored, wrote {out_path}")
    return out_path


def run_agentdojo(args: argparse.Namespace) -> Path | None:
    config = load_project_config()
    benchmark = config.benchmark
    agentdojo_cfg = benchmark["datasets"]["agentdojo"]

    model = args.model or "gpt-4o-mini"
    defense = args.defense or "baseline"
    suite = agentdojo_cfg["suite"]
    attack = agentdojo_cfg.get("attack", DEFAULT_ATTACK)
    user_tasks = agentdojo_cfg.get("pilot_user_tasks") if args.mode == "pilot" else None

    logdir = resolve_path(benchmark["output"]["raw_dir"]) / "agentdojo_logs" / str(int(time.time()))
    command = build_agentdojo_command(model, defense, suite, user_tasks, attack, logdir)

    if args.dry_run:
        print(" ".join(command))
        return None

    returncode = subprocess.call(command)
    if returncode != 0:
        raise RuntimeError(f"AgentDojo benchmark failed with exit code {returncode}")

    pipeline_name = map_model_to_agentdojo(model)
    rows = parse_agentdojo_logdir(logdir, pipeline_name=pipeline_name, suite_name=suite, defense=defense)

    raw_dir = resolve_path(benchmark["output"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"agentdojo_{int(time.time())}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} AgentDojo rows to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["custom", "agentdojo"], default="custom")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--model")
    parser.add_argument("--defense", choices=list(DEFENSES))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--categories", help="Comma-separated scenario categories to include (default: all).")
    parser.add_argument(
        "--attack-techniques", help="Comma-separated attack techniques to include (default: all)."
    )
    parser.add_argument("--severities", help="Comma-separated severities to include (default: all).")
    parser.add_argument(
        "--judge-model",
        help="Override the judge model from configs/benchmark.yaml (any configured model, any provider).",
    )
    parser.add_argument(
        "--run-group-id",
        help=(
            "Shared identifier (typically epoch seconds) for every model/defense combination launched from "
            "the same experiment submission, so the dashboard can group them as one run instead of one per "
            "model/defense pair. Defaults to this invocation's own start time if omitted."
        ),
    )
    parser.add_argument("--scenario-path")
    parser.add_argument("--dataset-label", default="custom")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dataset == "agentdojo":
        run_agentdojo(args)
        return
    out_path = run_custom(args)
    print(f"Wrote raw results to {out_path}")


if __name__ == "__main__":
    main()
