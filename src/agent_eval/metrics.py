from __future__ import annotations

import math
import re
from datetime import datetime

import pandas as pd

# run_group_id is epoch seconds (as a string) shared by every model/defense combination
# launched from the same "Run Experiment" submission (see runner.py's run_custom and
# dashboard/app.py's run_experiment callback), so the moment a *run* happened -- as opposed
# to the moment one particular model/defense subprocess call happened -- can be recovered
# directly from it. Raw filenames follow the older "<prefix>_<epoch-seconds>.jsonl" pattern
# and are used as a fallback for rows written before run_group_id existed.
_EPOCH_PATTERN = re.compile(r"(\d{9,})(?:\.jsonl)?$")


def extract_run_timestamp(token: str | None) -> str | None:
    if not token:
        return None
    match = _EPOCH_PATTERN.search(str(token))
    if not match:
        return None
    try:
        return datetime.fromtimestamp(int(match.group(1))).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, ValueError, OverflowError):
        return None


SUMMARY_COLUMNS = [
    "run_id",
    "run_group_id",
    "source_file",
    "run_timestamp",
    "model",
    "dataset",
    "scenario_id",
    "category",
    "attack_technique",
    "injection_position",
    "severity",
    "defense",
    "condition",
    "judge_model",
    "task_success",
    "attack_success",
    "robust_success",
    "keyword_task_success",
    "keyword_attack_success",
    "judge_confidence",
    "judge_rationale",
    "judge_error",
    "latency_seconds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "estimated_cost_usd",
    "judge_cost_usd",
    "total_cost_usd",
    "user_task",
    "untrusted_content",
    "model_output",
    "error",
    "error_type",
]


def normalize_results(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    for column in ["task_success", "attack_success", "robust_success"]:
        frame[column] = frame[column].fillna(False).astype(bool)
    # Rows written before run_group_id existed fall back to grouping by source_file (one
    # raw file per model/defense subprocess call), matching the old per-file behavior.
    frame["run_group_id"] = frame["run_group_id"].where(frame["run_group_id"].notna(), frame["source_file"])
    frame["run_timestamp"] = frame["run_group_id"].apply(extract_run_timestamp)
    return frame[SUMMARY_COLUMNS]


def aggregate_leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "defense",
                "benign_success",
                "attacked_success",
                "utility_drop",
                "attack_success_rate",
                "robust_success_rate",
                "avg_latency_seconds",
                "runs",
                "runs_benign",
                "runs_attacked",
            ]
        )

    grouped = summary.groupby(["model", "defense"], dropna=False)
    rows = []
    for (model, defense), group in grouped:
        benign = group[group["condition"] == "benign"]
        attacked = group[group["condition"] == "attacked"]
        benign_success = float(benign["task_success"].mean()) if len(benign) else math.nan
        attacked_success = float(attacked["task_success"].mean()) if len(attacked) else math.nan
        attack_success = float(attacked["attack_success"].mean()) if len(attacked) else math.nan
        robust_success = float(attacked["robust_success"].mean()) if len(attacked) else math.nan
        rows.append(
            {
                "model": model,
                "defense": defense,
                "benign_success": benign_success,
                "attacked_success": attacked_success,
                "utility_drop": benign_success - attacked_success,
                "attack_success_rate": attack_success,
                "robust_success_rate": robust_success,
                "avg_latency_seconds": float(group["latency_seconds"].fillna(0).mean()),
                "runs": int(len(group)),
                "runs_benign": int(len(benign)),
                "runs_attacked": int(len(attacked)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["robust_success_rate", "attack_success_rate"], ascending=[False, True]
    )


def summarize_runs(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct run_group_id (every model/defense combination launched from the
    same "Run Experiment" submission or CLI invocation shares one run_group_id), newest
    first, so it's possible to tell which rows in the combined summary came from which run
    and when -- the aggregated leaderboard alone collapses that distinction away."""
    columns = ["run_timestamp", "run_group_id", "dataset", "models", "defenses", "rows", "robust_success_rate"]
    if summary.empty or "run_group_id" not in summary.columns:
        return pd.DataFrame(columns=columns)

    rows = []
    for run_group_id, group in summary.groupby("run_group_id", dropna=False):
        attacked = group[group["condition"] == "attacked"]
        rows.append(
            {
                "run_timestamp": group["run_timestamp"].iloc[0],
                "run_group_id": run_group_id,
                "dataset": ", ".join(sorted(group["dataset"].dropna().unique())),
                "models": ", ".join(sorted(group["model"].dropna().unique())),
                "defenses": ", ".join(sorted(group["defense"].dropna().unique())),
                "rows": int(len(group)),
                "robust_success_rate": float(attacked["robust_success"].mean()) if len(attacked) else math.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("run_timestamp", ascending=False, na_position="last")
