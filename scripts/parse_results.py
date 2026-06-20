from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_eval.config import load_project_config, resolve_path
from agent_eval.metrics import aggregate_leaderboard, normalize_results


def read_raw_results(raw_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row.setdefault("source_file", path.name)
                    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir")
    parser.add_argument("--summary-out")
    parser.add_argument("--leaderboard-out")
    args = parser.parse_args()

    config = load_project_config().benchmark
    raw_dir = resolve_path(args.raw_dir or config["output"]["raw_dir"])
    summary_out = resolve_path(args.summary_out or config["output"]["summary_csv"])
    leaderboard_out = resolve_path(args.leaderboard_out or config["output"]["leaderboard_csv"])

    rows = read_raw_results(raw_dir)
    summary = normalize_results(rows)
    leaderboard = aggregate_leaderboard(summary)

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    leaderboard.to_csv(leaderboard_out, index=False)
    print(f"Wrote {len(summary)} rows to {summary_out}")
    print(f"Wrote leaderboard to {leaderboard_out}")


if __name__ == "__main__":
    main()
