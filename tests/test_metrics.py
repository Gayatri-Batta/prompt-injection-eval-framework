import pandas as pd

from agent_eval.metrics import aggregate_leaderboard, extract_run_timestamp, normalize_results, summarize_runs


def test_aggregate_leaderboard_rates():
    summary = pd.DataFrame(
        [
            {"model": "m1", "defense": "baseline", "condition": "benign", "task_success": True, "attack_success": False, "robust_success": True, "latency_seconds": 1},
            {"model": "m1", "defense": "baseline", "condition": "attacked", "task_success": False, "attack_success": True, "robust_success": False, "latency_seconds": 1},
            {"model": "m1", "defense": "baseline", "condition": "attacked", "task_success": True, "attack_success": False, "robust_success": True, "latency_seconds": 1},
        ]
    )
    out = aggregate_leaderboard(summary).iloc[0]
    assert out["benign_success"] == 1.0
    assert out["attacked_success"] == 0.5
    assert out["utility_drop"] == 0.5
    assert out["attack_success_rate"] == 0.5
    assert out["robust_success_rate"] == 0.5
    assert out["runs_benign"] == 1
    assert out["runs_attacked"] == 2


def test_aggregate_leaderboard_missing_benign_is_nan_not_zero():
    summary = pd.DataFrame(
        [
            {"model": "m1", "defense": "baseline", "condition": "attacked", "task_success": False, "attack_success": False, "robust_success": False, "latency_seconds": 1},
        ]
    )
    out = aggregate_leaderboard(summary).iloc[0]
    assert pd.isna(out["benign_success"])
    assert pd.isna(out["utility_drop"])
    assert out["runs_benign"] == 0
    assert out["runs_attacked"] == 1


def test_aggregate_leaderboard_missing_attacked_is_nan_not_zero():
    summary = pd.DataFrame(
        [
            {"model": "m1", "defense": "baseline", "condition": "benign", "task_success": True, "attack_success": False, "robust_success": True, "latency_seconds": 1},
        ]
    )
    out = aggregate_leaderboard(summary).iloc[0]
    assert pd.isna(out["attacked_success"])
    assert pd.isna(out["attack_success_rate"])
    assert pd.isna(out["robust_success_rate"])
    assert pd.isna(out["utility_drop"])
    assert out["runs_benign"] == 1
    assert out["runs_attacked"] == 0


def test_extract_run_timestamp_parses_epoch_from_filename():
    timestamp = extract_run_timestamp("custom_openai_1781918771.jsonl")
    assert timestamp is not None
    assert timestamp.startswith("20")  # a real calendar year, not None/garbage


def test_extract_run_timestamp_handles_missing_or_malformed_filenames():
    assert extract_run_timestamp(None) is None
    assert extract_run_timestamp("") is None
    assert extract_run_timestamp("not_a_timestamped_file.jsonl") is None


def test_extract_run_timestamp_parses_bare_epoch_string():
    # run_group_id is a bare epoch-seconds string (no filename wrapper), unlike source_file.
    timestamp = extract_run_timestamp("1781918771")
    assert timestamp is not None
    assert timestamp.startswith("20")


def test_normalize_results_derives_run_timestamp_from_source_file():
    frame = normalize_results([{"model": "m1", "source_file": "custom_openai_1781918771.jsonl"}])
    assert frame["run_timestamp"].iloc[0] is not None
    assert frame["source_file"].iloc[0] == "custom_openai_1781918771.jsonl"


def test_normalize_results_falls_back_to_source_file_when_run_group_id_missing():
    # Rows written before run_group_id existed shouldn't lose per-run grouping entirely.
    frame = normalize_results([{"model": "m1", "source_file": "custom_openai_1781918771.jsonl"}])
    assert frame["run_group_id"].iloc[0] == "custom_openai_1781918771.jsonl"


def test_normalize_results_prefers_explicit_run_group_id_over_source_file():
    frame = normalize_results(
        [{"model": "m1", "source_file": "custom_openai_111.jsonl", "run_group_id": "1781918771"}]
    )
    assert frame["run_group_id"].iloc[0] == "1781918771"
    assert frame["run_timestamp"].iloc[0] is not None


def test_summarize_runs_groups_by_run_group_id():
    # Two source_files sharing one run_group_id simulate one "Run Experiment" click that
    # launched two model/defense combinations as separate subprocess calls/files -- they
    # should still collapse into a single run in the history.
    summary = pd.DataFrame(
        [
            {"source_file": "run_a1.jsonl", "run_group_id": "100", "run_timestamp": "2026-01-01 10:00:00", "dataset": "custom", "model": "m1", "defense": "baseline", "condition": "attacked", "robust_success": True},
            {"source_file": "run_a2.jsonl", "run_group_id": "100", "run_timestamp": "2026-01-01 10:00:00", "dataset": "custom", "model": "m1", "defense": "baseline", "condition": "attacked", "robust_success": False},
            {"source_file": "run_b.jsonl", "run_group_id": "200", "run_timestamp": "2026-01-02 10:00:00", "dataset": "custom", "model": "m2", "defense": "baseline", "condition": "benign", "robust_success": True},
        ]
    )
    runs = summarize_runs(summary)
    assert len(runs) == 2
    # newest run first
    assert runs.iloc[0]["run_group_id"] == "200"
    run_a = runs[runs["run_group_id"] == "100"].iloc[0]
    assert run_a["rows"] == 2
    assert run_a["robust_success_rate"] == 0.5
    assert run_a["models"] == "m1"


def test_summarize_runs_empty_summary_returns_empty_frame():
    assert summarize_runs(pd.DataFrame()).empty
