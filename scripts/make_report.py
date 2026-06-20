from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agent_eval.config import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard", default="results/leaderboard.csv")
    parser.add_argument("--out", default="reports/final_report.md")
    args = parser.parse_args()

    leaderboard_path = resolve_path(args.leaderboard)
    out_path = resolve_path(args.out)
    leaderboard = pd.read_csv(leaderboard_path) if leaderboard_path.exists() else pd.DataFrame()

    table = leaderboard.to_markdown(index=False) if not leaderboard.empty else "_No results yet._"
    content = f"""# Prompt Injection Safety Evaluation Report

## Research Question

Which GPT model and defense setup gives the best tradeoff between useful task completion and resistance to indirect prompt injection in workplace-agent tasks?

## Method

The platform evaluates workplace-agent scenarios under benign and attacked conditions. Each run records task success, attack success, robust success, latency, and model output. AgentDojo can be run separately as a validated public benchmark; the included custom dataset provides reproducible workplace scenarios.

## Leaderboard

{table}

## Notes

- `robust_success_rate` is the most important safety/usefulness metric.
- All benchmark runs use real OpenAI API calls and require `OPENAI_API_KEY`.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
