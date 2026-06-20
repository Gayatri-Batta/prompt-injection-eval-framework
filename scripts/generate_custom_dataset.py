from __future__ import annotations

import argparse

from agent_eval.config import resolve_path
from agent_eval.custom_dataset import generate_scenarios, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Cap on number of scenarios. Defaults to the full task x attack-technique cross product.",
    )
    parser.add_argument("--out", default="data/custom_scenarios.jsonl")
    args = parser.parse_args()

    scenarios = generate_scenarios(args.count)
    out_path = resolve_path(args.out)
    write_jsonl(scenarios, out_path)
    print(f"Wrote {len(scenarios)} scenarios to {out_path}")


if __name__ == "__main__":
    main()
