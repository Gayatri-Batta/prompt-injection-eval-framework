from __future__ import annotations

from pathlib import Path

import duckdb


def build_database(summary_csv: str | Path, leaderboard_csv: str | Path, db_path: str | Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        con.execute("CREATE OR REPLACE TABLE runs AS SELECT * FROM read_csv_auto(?)", [str(summary_csv)])
        con.execute(
            "CREATE OR REPLACE TABLE leaderboard AS SELECT * FROM read_csv_auto(?)",
            [str(leaderboard_csv)],
        )


def query_dataframe(db_path: str | Path, query: str):
    with duckdb.connect(str(db_path)) as con:
        return con.execute(query).df()
