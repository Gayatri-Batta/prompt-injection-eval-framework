from pathlib import Path

import pytest

from agent_eval.agentdojo_adapter import (
    build_agentdojo_command,
    map_defense_to_agentdojo,
    map_model_to_agentdojo,
    parse_agentdojo_logdir,
)

FIXTURES = Path(__file__).parent / "fixtures" / "agentdojo_logs"


def test_map_model_to_agentdojo_known_ids():
    assert map_model_to_agentdojo("gpt-4o-mini") == "gpt-4o-mini-2024-07-18"
    assert map_model_to_agentdojo("gpt-4o") == "gpt-4o-2024-05-13"
    assert map_model_to_agentdojo("gpt-3.5-turbo") == "gpt-3.5-turbo-0125"


def test_map_model_to_agentdojo_unknown_id_raises():
    with pytest.raises(ValueError):
        map_model_to_agentdojo("not-a-real-model")


def test_map_defense_to_agentdojo():
    assert map_defense_to_agentdojo("baseline") is None
    assert map_defense_to_agentdojo("instruction_boundary") == "tool_filter"
    with pytest.raises(ValueError):
        map_defense_to_agentdojo("unknown_defense")


def test_build_agentdojo_command_includes_attack_and_defense():
    command = build_agentdojo_command(
        model="gpt-4o-mini",
        defense="instruction_boundary",
        suite="slack",
        user_tasks=["user_task_0", "user_task_1"],
        attack="important_instructions",
        logdir=Path("/tmp/logs"),
    )
    assert "--model" in command and "gpt-4o-mini-2024-07-18" in command
    assert "--defense" in command and "tool_filter" in command
    assert "--attack" in command and "important_instructions" in command
    assert command.count("-ut") == 2


def test_build_agentdojo_command_omits_defense_flag_for_baseline():
    command = build_agentdojo_command(
        model="gpt-4o-mini",
        defense="baseline",
        suite="slack",
        user_tasks=None,
        attack="important_instructions",
        logdir=Path("/tmp/logs"),
    )
    assert "--defense" not in command


def test_parse_agentdojo_logdir_maps_rows_correctly():
    rows = parse_agentdojo_logdir(
        FIXTURES, pipeline_name="gpt-4o-mini-2024-07-18", suite_name="slack", defense="baseline"
    )
    assert len(rows) == 3

    by_scenario = {r["scenario_id"]: r for r in rows}

    benign = by_scenario["user_task_0__none"]
    assert benign["condition"] == "benign"
    assert benign["task_success"] is True
    assert benign["attack_success"] is None
    assert benign["robust_success"] is True
    assert benign["dataset"] == "agentdojo"
    assert benign["defense"] == "baseline"

    attacked_success = by_scenario["user_task_0__injection_task_0"]
    assert attacked_success["condition"] == "attacked"
    assert attacked_success["task_success"] is True
    assert attacked_success["attack_success"] is True  # security=false -> attack succeeded
    assert attacked_success["robust_success"] is False

    attacked_failure = by_scenario["user_task_1__injection_task_0"]
    assert attacked_failure["condition"] == "attacked"
    assert attacked_failure["attack_success"] is False  # security=true -> attack failed
    assert attacked_failure["robust_success"] is True


def test_parse_agentdojo_logdir_missing_dir_returns_empty():
    rows = parse_agentdojo_logdir(
        Path("/does/not/exist"), pipeline_name="x", suite_name="y", defense="baseline"
    )
    assert rows == []
