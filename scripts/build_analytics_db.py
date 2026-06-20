from __future__ import annotations

import argparse

from agent_eval.config import load_project_config, resolve_path
from agent_eval.duckdb_analysis import build_database


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary")
    parser.add_argument("--leaderboard")
    parser.add_argument("--db")
    args = parser.parse_args()

    config = load_project_config().benchmark["output"]
    summary = resolve_path(args.summary or config["summary_csv"])
    leaderboard = resolve_path(args.leaderboard or config["leaderboard_csv"])
    db = resolve_path(args.db or config["duckdb_path"])
    build_database(summary, leaderboard, db)
    print(f"Wrote DuckDB analytics database to {db}")


if __name__ == "__main__":
    main()
