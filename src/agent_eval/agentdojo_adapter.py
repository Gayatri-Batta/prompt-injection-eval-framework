from __future__ import annotations

import json
from pathlib import Path

# AgentDojo's CLI (agentdojo.scripts.benchmark) expects dated model identifiers, not the bare
# ids we use in configs/models.yaml. This mapping is current as of agentdojo==0.1.35 -- if a
# newer version is installed, diff this against `python -m agentdojo.scripts.benchmark --help`
# before trusting it, since the registry can change between releases.
MODEL_TO_AGENTDOJO = {
    "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o": "gpt-4o-2024-05-13",
}

# AgentDojo's defense vocabulary (tool_filter, transformers_pi_detector, etc.) is not the same
# taxonomy as our custom-dataset defenses (baseline, instruction_boundary, spotlighting,
# sandwich). Where the technique genuinely matches an AgentDojo defense (spotlighting,
# repeat_user_prompt/"sandwich") we map to it directly so the comparison is meaningful;
# instruction_boundary has no exact AgentDojo equivalent so it's mapped to the closest
# available defense (tool_filter) -- the dataset column still distinguishes which benchmark
# produced the row, so the two are never conflated in the leaderboard.
DEFENSE_TO_AGENTDOJO = {
    "baseline": None,
    "instruction_boundary": "tool_filter",
    "spotlighting": "spotlighting_with_delimiting",
    "sandwich": "repeat_user_prompt",
}

DEFAULT_ATTACK = "important_instructions"


def map_model_to_agentdojo(model_id: str) -> str:
    try:
        return MODEL_TO_AGENTDOJO[model_id]
    except KeyError as exc:
        raise ValueError(
            f"No known AgentDojo model id for '{model_id}'. Add it to MODEL_TO_AGENTDOJO "
            "in agentdojo_adapter.py (check `python -m agentdojo.scripts.benchmark --help` "
            "for the currently valid --model choices)."
        ) from exc


def map_defense_to_agentdojo(defense: str) -> str | None:
    if defense not in DEFENSE_TO_AGENTDOJO:
        raise ValueError(
            f"No known AgentDojo defense mapping for '{defense}'. Add it to DEFENSE_TO_AGENTDOJO "
            "in agentdojo_adapter.py."
        )
    return DEFENSE_TO_AGENTDOJO[defense]


def build_agentdojo_command(
    model: str,
    defense: str,
    suite: str,
    user_tasks: list[str] | None,
    attack: str | None,
    logdir: Path,
) -> list[str]:
    agentdojo_model = map_model_to_agentdojo(model)
    agentdojo_defense = map_defense_to_agentdojo(defense)
    command = [
        "python",
        "-m",
        "agentdojo.scripts.benchmark",
        "-s",
        suite,
        "--model",
        agentdojo_model,
        "--logdir",
        str(logdir),
    ]
    if agentdojo_defense:
        command.extend(["--defense", agentdojo_defense])
    if attack:
        command.extend(["--attack", attack])
    if user_tasks:
        for user_task in user_tasks:
            command.extend(["-ut", user_task])
    return command


def parse_agentdojo_logdir(logdir: Path, pipeline_name: str, suite_name: str, defense: str) -> list[dict]:
    """Parse AgentDojo's per-task result JSON files (TaskResults schema) into rows compatible
    with agent_eval.metrics.SUMMARY_COLUMNS. AgentDojo writes one JSON file per
    (pipeline_name, suite_name, user_task_id, attack_type, injection_task_id) combination at
    `{logdir}/{pipeline_name}/{suite_name}/{user_task_id}/{attack_type}/{injection_task_id or 'none'}.json`.
    """
    base = Path(logdir) / pipeline_name / suite_name
    rows: list[dict] = []
    if not base.exists():
        return rows

    for result_path in sorted(base.glob("**/*.json")):
        with result_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        user_task_id = data.get("user_task_id")
        injection_task_id = data.get("injection_task_id")
        attack_type = data.get("attack_type")
        has_injection = injection_task_id not in (None, "none")
        condition = "attacked" if has_injection else "benign"

        utility = bool(data.get("utility"))
        security = data.get("security")
        attack_success = (not bool(security)) if (has_injection and security is not None) else None
        robust_success = (utility and not attack_success) if attack_success is not None else utility

        messages = data.get("messages") or []
        assistant_texts = [
            m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content")
        ]

        rows.append(
            {
                "run_id": f"{pipeline_name}__{suite_name}__{user_task_id}__{injection_task_id or 'none'}",
                "model": pipeline_name,
                "agentdojo_model_id": pipeline_name,
                "dataset": "agentdojo",
                "scenario_id": f"{user_task_id}__{injection_task_id or 'none'}",
                "category": suite_name,
                "attack_technique": attack_type,
                "injection_position": None,
                "severity": None,
                "defense": defense,
                "condition": condition,
                "task_success": utility,
                "attack_success": bool(attack_success) if attack_success is not None else None,
                "robust_success": bool(robust_success),
                "keyword_task_success": None,
                "keyword_attack_success": None,
                "judge_confidence": None,
                "judge_rationale": None,
                "judge_error": None,
                "latency_seconds": data.get("duration"),
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "estimated_cost_usd": None,
                "judge_cost_usd": None,
                "total_cost_usd": None,
                "user_task": user_task_id,
                "untrusted_content": json.dumps(data.get("injections") or {}, ensure_ascii=False),
                "model_output": "\n".join(assistant_texts),
                "error": data.get("error"),
                "error_type": None,
                "attack_type": attack_type,
                "agentdojo_package_version": data.get("agentdojo_package_version"),
                "evaluation_timestamp": data.get("evaluation_timestamp"),
            }
        )
    return rows
