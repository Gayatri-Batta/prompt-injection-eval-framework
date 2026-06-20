from __future__ import annotations

import base64
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from agent_eval.config import get_model_pricing, load_project_config, missing_api_key_models, resolve_path
from agent_eval.custom_dataset import (
    EXAMPLE_SCENARIO_ROW,
    RECOMMENDED_SCENARIO_FIELDS,
    REQUIRED_SCENARIO_FIELDS,
    attack_technique_summary,
    category_counts,
    filter_scenario_rows,
    parse_scenario_text,
    read_jsonl,
    severity_counts,
    validate_scenario_rows,
)
from agent_eval.defenses import get_defense_info
from agent_eval.metrics import aggregate_leaderboard, summarize_runs
from agent_eval.runner import compute_cost_usd
from agent_eval.plotting import category_attack_rate, leaderboard_bar, task_heatmap, utility_vs_attack


SUMMARY_PATH = ROOT / "results" / "summary.csv"
UPLOAD_PATH = ROOT / "data" / "uploaded_scenarios.jsonl"
SELECTED_PATH = ROOT / "data" / "selected_scenarios.jsonl"

_PROJECT_CONFIG = load_project_config()
DEFAULT_MODELS = [m["id"] for m in _PROJECT_CONFIG.models]
DEFAULT_DEFENSES = _PROJECT_CONFIG.benchmark["defenses"]
JUDGING_CFG = _PROJECT_CONFIG.benchmark.get("judging") or {}


def _provider_label(provider_id: str) -> str:
    info = _PROJECT_CONFIG.providers.get(provider_id) or {}
    return info.get("label", provider_id)


def model_dropdown_options() -> list[dict]:
    options = []
    for model in _PROJECT_CONFIG.models:
        model_id = model["id"]
        provider_label = _provider_label(model.get("provider", ""))
        options.append({"label": f"{model_id} ({provider_label})", "value": model_id})
    return options


def defense_dropdown_options() -> list[dict]:
    return [{"label": get_defense_info(defense_id)["label"], "value": defense_id} for defense_id in DEFAULT_DEFENSES]


def _example_csv_snippet() -> str:
    row = EXAMPLE_SCENARIO_ROW
    header = ["scenario_id", "category", "user_task", "benign_content", "expected_keywords"]
    values = [
        row["scenario_id"],
        row["category"],
        row["user_task"],
        row["benign_content"],
        ";".join(row["expected_keywords"]),
    ]
    quoted = [f'"{v}"' if "," in v else v for v in values]
    return ",".join(header) + "\n" + ",".join(quoted)


def _default_scenario_count() -> int:
    try:
        path = resolve_path(_PROJECT_CONFIG.benchmark["datasets"]["custom"]["path"])
        return max(len(read_jsonl(path)), 30)
    except (FileNotFoundError, KeyError):
        return 30


DEFAULT_SCENARIO_LIMIT_MAX = _default_scenario_count()

# Tracks the highest run-button n_clicks value that has actually launched a real run, so a
# spurious re-fire of the callback (e.g. a component remount resetting n_clicks to its literal
# default) can never launch another one. See run_experiment().
_LAST_HANDLED_RUN_CLICKS = 0

# Rough per-call token averages pulled from results/summary.csv's historical real-run data, used
# only to ballpark a pre-run cost estimate -- actual usage depends on the specific content length.
_AVG_PROMPT_TOKENS = 157
_AVG_COMPLETION_TOKENS = 37

COLORS = {
    "bg": "#f6f8fc",
    "panel": "#ffffff",
    "ink": "#172033",
    "muted": "#4b5563",
    "line": "#d7deea",
    "blue": "#2563eb",
    "green": "#059669",
    "red": "#dc2626",
    "amber": "#b45309",
}


if load_dotenv is not None:
    load_dotenv(ROOT / ".env")


def load_data() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_PATH) if SUMMARY_PATH.exists() else pd.DataFrame()
    return summary


def dropdown_options(values):
    return [{"label": str(value), "value": value} for value in values]


def category_dropdown_options(scenarios: list[dict]) -> list[dict]:
    counts = category_counts(scenarios)
    return [{"label": category, "value": category} for category in sorted(counts)]


def attack_technique_dropdown_options(scenarios: list[dict]) -> list[dict]:
    summary = attack_technique_summary(scenarios)
    return [{"label": technique, "value": technique} for technique in sorted(summary)]


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "unknown": 3}


def severity_dropdown_options(scenarios: list[dict]) -> list[dict]:
    counts = severity_counts(scenarios)
    return [
        {"label": severity, "value": severity}
        for severity in sorted(counts, key=lambda item: _SEVERITY_ORDER.get(item, 9))
    ]


def pct(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value * 100:.1f}%"


def _format_run_timestamp(raw: str) -> str | None:
    try:
        dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None
    hour = dt.strftime("%I").lstrip("0") or "12"
    return f"{dt.strftime('%B %d')}, {hour}:{dt.strftime('%M %p')}"


def card(label: str, value: str, detail: str, accent: str):
    return html.Div(
        [
            html.Div(label, className="kpi-label"),
            html.Div(value, style={"fontSize": "30px", "fontWeight": 850, "color": accent}),
            html.Div(detail, className="muted"),
        ],
        className="card",
    )


def run_command(args: list[str]) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=1200,
    )
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part.strip())
    if completed.returncode != 0:
        raise RuntimeError(output or f"Command failed: {' '.join(args)}")
    return output


def save_uploaded_dataset(contents: str, filename: str) -> int:
    if not contents:
        raise ValueError("Upload a dataset file before choosing uploaded dataset.")
    suffix = Path(filename or "").suffix.lower()
    if suffix not in (".jsonl", ".json", ".csv", ""):
        raise ValueError(f"Unsupported file type '{suffix}'. Upload a .jsonl, .json, or .csv file.")

    _, encoded = contents.split(",", 1)
    decoded = base64.b64decode(encoded).decode("utf-8")
    if not decoded.strip():
        raise ValueError("Uploaded file is empty.")

    rows = parse_scenario_text(filename or "scenarios.jsonl", decoded)
    errors = validate_scenario_rows(rows)
    if errors:
        raise ValueError(
            "Dataset failed validation:\n"
            + "\n".join(f"  - {error}" for error in errors)
            + "\n\nSee the \"Dataset format\" guide above for the expected schema."
        )

    # Re-serialize (rather than writing the raw uploaded bytes) so the saved file is
    # guaranteed to be one valid JSON object per line, matching what was actually validated.
    write_scenario_rows_jsonl(rows, UPLOAD_PATH)
    return len(rows)


def write_scenario_rows_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


summary_df = load_data()
models = sorted(set(DEFAULT_MODELS) | set(summary_df["model"].dropna().unique())) if not summary_df.empty else DEFAULT_MODELS
defenses = sorted(set(DEFAULT_DEFENSES) | set(summary_df["defense"].dropna().unique())) if not summary_df.empty else DEFAULT_DEFENSES
datasets = sorted(set(["custom", "uploaded"]) | set(summary_df["dataset"].dropna().unique())) if not summary_df.empty else ["custom", "uploaded"]

_run_history_initial = summarize_runs(summary_df) if not summary_df.empty else pd.DataFrame()
run_filter_options = [
    {
        "label": f"{row['run_timestamp'] or 'unknown time'} — {row['dataset'] or 'n/a'} — {row['models'] or 'n/a'}",
        "value": row["run_group_id"],
    }
    for _, row in _run_history_initial.iterrows()
]
run_filter_values = [opt["value"] for opt in run_filter_options]

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Prompt Injection Safety Lab"

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { margin: 0; background: #f6f8fc; }
            .app { font-family: Inter, Segoe UI, Arial, sans-serif; color: #172033; }
            .hero { background: linear-gradient(135deg, #111827 0%, #1d4ed8 100%); color: white; padding: 26px 34px; display: flex; justify-content: space-between; gap: 24px; align-items: center; }
            .hero h1 { margin: 6px 0; font-size: 30px; letter-spacing: 0; }
            .shell { display: grid; grid-template-columns: 292px 1fr; min-height: calc(100vh - 112px); }
            .sidebar { background: white; border-right: 1px solid #d7deea; padding: 20px; }
            .main { padding: 20px; overflow: hidden; }
            .filter-label { display: block; margin: 15px 0 6px; font-size: 13px; font-weight: 850; color: #4b5563; text-transform: uppercase; }
            .card { background: white; border: 1px solid #d7deea; border-radius: 8px; padding: 16px; box-shadow: 0 8px 22px rgba(20, 32, 55, 0.05); }
            .kpi-label { font-size: 13px; color: #4b5563; font-weight: 800; }
            .muted { font-size: 13px; color: #4b5563; margin-top: 6px; }
            .button-primary { background: #2563eb; color: white; border: 0; border-radius: 8px; padding: 12px 16px; font-weight: 850; cursor: pointer; }
            .button-secondary { background: #e5e7eb; color: #172033; border: 0; border-radius: 8px; padding: 12px 16px; font-weight: 850; cursor: pointer; margin-left: 10px; }
            button:disabled { opacity: 0.55; cursor: wait; }
            .notice { border-radius: 8px; padding: 12px; font-size: 13px; background: white; border: 1px solid #d7deea; }
            .dash-table-container .dash-spreadsheet-container { border-radius: 8px; }
            .tab-container { border-bottom: 1px solid #d7deea; }
            .tab { font-weight: 700; color: #4b5563; background: #eef1f7; transition: background 0.15s ease, color 0.15s ease; }
            .tab:hover { background: #e3e8f3; color: #172033; }
            .tab--selected, .tab--selected:hover {
                font-weight: 850 !important;
                color: #2563eb !important;
                background: white !important;
                border-top: 3px solid #2563eb !important;
                box-shadow: 0 -2px 10px rgba(37, 99, 235, 0.12);
            }
            .status-badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 850; text-transform: uppercase; letter-spacing: 0.03em; }
            .status-badge--idle { background: #eef1f7; color: #4b5563; }
            .status-badge--running { background: #dbeafe; color: #2563eb; }
            .status-badge--done { background: #d1fae5; color: #059669; }
            .status-badge--error { background: #fee2e2; color: #dc2626; }
            .subtab-container { border-bottom: 1px solid #d7deea; margin-top: 12px; }
            .subtab { font-weight: 700; font-size: 12px; color: #4b5563; background: transparent; padding: 8px 14px !important; border-top: none !important; }
            .subtab:hover { color: #2563eb; }
            .subtab--selected, .subtab--selected:hover {
                font-weight: 850 !important;
                color: #2563eb !important;
                background: transparent !important;
                border-top: none !important;
                border-bottom: 2px solid #2563eb !important;
                box-shadow: none !important;
            }
            .progress-step { display: flex; align-items: flex-start; gap: 10px; padding: 10px 4px; border-bottom: 1px solid #eef1f7; }
            .progress-step:last-child { border-bottom: none; }
            .progress-step-icon { flex-shrink: 0; width: 20px; height: 20px; border-radius: 999px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 850; color: white; }
            .progress-step-icon--done { background: #059669; }
            .progress-step-icon--running { background: #2563eb; }
            .progress-step-icon--error { background: #dc2626; }
            .progress-step-label { font-size: 13px; font-weight: 700; color: #172033; }
            .progress-step-detail { font-size: 13px; color: #4b5563; margin-top: 2px; }
            .logs-box { background: #f8fafc; color: #172033; border: 1px solid #d7deea; border-radius: 8px; padding: 16px; min-height: 360px; max-height: 480px; overflow-y: auto; white-space: pre-wrap; font-family: "Consolas", "SFMono-Regular", Menlo, monospace; font-size: 12px; line-height: 1.5; }
            .config-section { border: 1px solid #d7deea; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }
            .config-section:last-of-type { margin-bottom: 0; }
            .config-section-summary { cursor: pointer; padding: 12px 16px; font-weight: 850; font-size: 15px; color: #172033; background: #f8fafc; list-style: none; display: flex; align-items: center; gap: 10px; user-select: none; }
            .config-section-summary:hover { background: #eef1f7; }
            .config-section-number { display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px; border-radius: 999px; background: #2563eb; color: white; font-size: 12px; font-weight: 850; flex-shrink: 0; }
            .config-section-summary-text { font-size: 12px; font-weight: 500; color: #4b5563; text-transform: none; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; min-width: 0; }
            .config-section-chevron { margin-left: auto; color: #4b5563; font-size: 12px; transition: transform 0.15s ease; flex-shrink: 0; }
            .config-section-chevron--open { transform: rotate(90deg); }
            .config-section-body { padding: 16px; border-top: 1px solid #d7deea; }
            .run-summary-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


app.layout = html.Div(
    [
        html.Div(
            [
                html.Div("Agent Safety Analytics", style={"fontSize": "13px", "color": "#bfdbfe", "fontWeight": 800}),
                html.H1("Prompt Injection Safety Lab"),
                html.Div(
                    "Run real model evaluations across OpenAI, Groq, and Anthropic on generated or uploaded workplace prompt-injection datasets.",
                    style={"color": "#dbeafe", "maxWidth": "780px"},
                ),
            ],
            className="hero",
        ),
        html.Div(
            [
                html.Aside(
                    [
                        html.Div(
                            [
                                html.H3("Results Filters", style={"marginTop": 0}),
                                html.Label("Model", className="filter-label"),
                                dcc.Dropdown(id="model-filter", options=dropdown_options(models), value=models, multi=True),
                                html.Label("Defense", className="filter-label"),
                                dcc.Dropdown(id="defense-filter", options=dropdown_options(defenses), value=defenses, multi=True),
                                html.Label("Dataset", className="filter-label"),
                                dcc.Dropdown(id="dataset-filter", options=dropdown_options(datasets), value=datasets, multi=True),
                                html.Label("Run", className="filter-label"),
                                dcc.Dropdown(
                                    id="run-filter",
                                    options=run_filter_options,
                                    value=run_filter_values,
                                    multi=True,
                                    placeholder="All runs",
                                ),
                            ],
                            id="results-filters-section",
                            style={"display": "none"},
                        ),
                    ],
                    className="sidebar",
                ),
                html.Main(
                    [
                        dcc.Store(id="run-completed-store"),
                        html.Div(id="kpi-running-banner", className="notice", style={"display": "none", "marginBottom": "10px"}),
                        html.Div(id="kpi-run-status", className="muted", style={"marginBottom": "10px"}),
                        html.Div(
                            [
                                html.Div(id="kpi-runs"),
                                html.Div(id="kpi-robust"),
                                html.Div(id="kpi-attack"),
                                html.Div(id="kpi-utility"),
                                html.Div(id="kpi-cost"),
                            ],
                            style={"display": "grid", "gridTemplateColumns": "repeat(5, minmax(0, 1fr))", "gap": "14px"},
                        ),
                        dcc.Tabs(
                            id="tabs",
                            value="experiment",
                            children=[
                                dcc.Tab(label="Run Experiment", value="experiment"),
                                dcc.Tab(label="Overview", value="overview"),
                                dcc.Tab(label="Scenario Analysis", value="scenarios"),
                                dcc.Tab(label="Run Drilldown", value="runs"),
                            ],
                            style={"marginTop": "18px"},
                        ),
                        html.Div(id="tab-content"),
                    ],
                    className="main",
                ),
            ],
            className="shell",
        ),
    ],
    className="app",
)


@app.callback(
    Output("results-filters-section", "style"),
    Input("tabs", "value"),
)
def toggle_results_filters(tab):
    if tab == "experiment":
        return {"display": "none"}
    return {"display": "block"}


def _apply_sidebar_filters(summary, selected_models, selected_defenses, selected_datasets, selected_runs):
    if selected_models:
        summary = summary[summary["model"].isin(selected_models)]
    if selected_defenses:
        summary = summary[summary["defense"].isin(selected_defenses)]
    if selected_datasets:
        summary = summary[summary["dataset"].isin(selected_datasets)]
    if selected_runs and "run_group_id" in summary.columns:
        summary = summary[summary["run_group_id"].isin(selected_runs)]
    return summary


def _leaderboard_charts(summary_subset: pd.DataFrame):
    if summary_subset.empty:
        return html.Div("No rows match this selection.", className="muted")
    leaderboard = aggregate_leaderboard(summary_subset)
    return html.Div(
        [
            html.Div(
                [
                    dcc.Graph(figure=leaderboard_bar(leaderboard), config={"displaylogo": False}),
                    dcc.Graph(figure=utility_vs_attack(leaderboard), config={"displaylogo": False}),
                ],
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            ),
            html.Div(
                "N/A means no rows were collected for that condition (not that the model failed every task).",
                className="muted",
                style={"marginTop": "10px"},
            ),
        ]
    )


def _run_leaderboard_table(summary_subset: pd.DataFrame):
    """Model x defense breakdown table for one run's rows. Includes `dataset` as a plain
    column (constant within a run, since one Run Experiment click uses a single dataset) so
    each accordion section is self-contained without needing a separate date/time column --
    the run's date/time is already the accordion header."""
    if summary_subset.empty:
        return html.Div("No rows for this run.", className="muted")
    leaderboard = aggregate_leaderboard(summary_subset)
    leaderboard_display = leaderboard.copy()
    for column in ["benign_success", "attacked_success", "utility_drop", "attack_success_rate", "robust_success_rate"]:
        leaderboard_display[column] = leaderboard_display[column].apply(pct)
    leaderboard_display["dataset"] = summary_subset["dataset"].iloc[0]
    leaderboard_cols = [
        "model", "dataset", "defense", "benign_success", "attacked_success", "utility_drop",
        "attack_success_rate", "robust_success_rate", "runs_benign", "runs_attacked",
    ]
    return dash_table.DataTable(
        data=leaderboard_display[[c for c in leaderboard_cols if c in leaderboard_display.columns]].to_dict("records"),
        columns=[{"name": col, "id": col} for col in leaderboard_cols if col in leaderboard_display.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#eef2ff", "fontWeight": "800", "border": "0"},
        style_cell={"textAlign": "left", "fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "12px", "padding": "10px", "border": f"1px solid {COLORS['line']}"},
    )


# Shared across the header row and every accordion row so the date/time, dataset, rows, and
# task-success-under-attack values line up as real table columns instead of being mentioned
# inline next to the date as free text.
_RUN_HISTORY_GRID_COLUMNS = "220px 1fr 100px 210px"


def _run_history_header_row():
    return html.Div(
        [
            html.Span("Date / Time"),
            html.Span("Dataset"),
            html.Span("Rows"),
            html.Span("Task Success Under Attack"),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": _RUN_HISTORY_GRID_COLUMNS,
            "padding": "4px 14px",
            "fontSize": "12px",
            "fontWeight": 800,
            "color": COLORS["muted"],
            "textTransform": "uppercase",
        },
    )


def _run_accordion_section(run_row, run_summary: pd.DataFrame, is_open: bool):
    timestamp = run_row["run_timestamp"] or "Unknown time"
    return html.Details(
        [
            html.Summary(
                html.Div(
                    [
                        html.Span(timestamp, style={"fontWeight": 800}),
                        html.Span(run_row["dataset"] or "n/a"),
                        html.Span(str(run_row["rows"])),
                        html.Span(pct(run_row["robust_success_rate"])),
                    ],
                    style={"display": "grid", "gridTemplateColumns": _RUN_HISTORY_GRID_COLUMNS, "alignItems": "center"},
                ),
                style={
                    "cursor": "pointer",
                    "padding": "10px 14px",
                    "background": "#eef2ff",
                    "borderRadius": "6px",
                    "listStyle": "none",
                },
            ),
            html.Div(_run_leaderboard_table(run_summary), style={"padding": "12px 4px 4px"}),
        ],
        open=is_open,
        style={"border": f"1px solid {COLORS['line']}", "borderRadius": "8px", "marginBottom": "10px", "padding": "2px"},
    )


@app.callback(
    Output("kpi-runs", "children"),
    Output("kpi-robust", "children"),
    Output("kpi-attack", "children"),
    Output("kpi-utility", "children"),
    Output("kpi-cost", "children"),
    Output("kpi-run-status", "children"),
    Output("tab-content", "children"),
    Input("model-filter", "value"),
    Input("defense-filter", "value"),
    Input("dataset-filter", "value"),
    Input("run-filter", "value"),
    Input("tabs", "value"),
    Input("run-completed-store", "data"),
)
def update_dashboard(selected_models, selected_defenses, selected_datasets, selected_runs, tab, _run_completed_token):
    # A run finishing writes to run-completed-store purely so the KPI cards/timestamp refresh
    # automatically -- it must never rebuild the experiment tab's own content. Rebuilding it would
    # recreate run-button/refresh-button as fresh component instances, and remounting an
    # Input-bound component can re-fire its callback despite prevent_initial_call=True, which
    # would launch another real run, which completes and writes run-completed-store again --
    # an infinite loop of real model API calls. See incident notes for 2026-06-20.
    skip_tab_content = ctx.triggered_id == "run-completed-store" and tab == "experiment"

    summary = load_data()
    summary = _apply_sidebar_filters(summary, selected_models, selected_defenses, selected_datasets, selected_runs)

    if summary.empty:
        return (
            card("Evaluations Completed", "0", "No rows yet", COLORS["blue"]),
            card("Task Success Under Attack", "N/A", "Task succeeds and attack fails", COLORS["green"]),
            card("Attack Success", "N/A", "Lower is better", COLORS["red"]),
            card("Utility Change", "N/A", "Benign minus attacked utility", COLORS["amber"]),
            card("Est. Cost", "N/A", "Combined target and judge model spend", COLORS["amber"]),
            "No completed runs yet.",
            no_update if skip_tab_content else experiment_panel(),
        )

    run_status = "Latest completed run -- unknown time"
    if "run_timestamp" in summary.columns and summary["run_timestamp"].notna().any():
        formatted = _format_run_timestamp(summary["run_timestamp"].dropna().max())
        if formatted:
            run_status = f"Latest completed run · {formatted}"

    attacked = summary[summary["condition"] == "attacked"]
    benign = summary[summary["condition"] == "benign"]
    robust = float(attacked["robust_success"].mean()) if len(attacked) else math.nan
    attack = float(attacked["attack_success"].mean()) if len(attacked) else math.nan
    benign_success = float(benign["task_success"].mean()) if len(benign) else math.nan
    attacked_success = float(attacked["task_success"].mean()) if len(attacked) else math.nan
    utility_drop = benign_success - attacked_success

    if "total_cost_usd" in summary.columns and summary["total_cost_usd"].notna().any():
        total_cost_display = f"${summary['total_cost_usd'].sum():.4f}"
    else:
        total_cost_display = "N/A"

    if tab == "experiment":
        content = no_update if skip_tab_content else experiment_panel()
    elif tab == "overview":
        run_history = summarize_runs(summary)
        has_run_tracking = not run_history.empty and "run_group_id" in summary.columns

        if has_run_tracking:
            latest = run_history.iloc[0]
            latest_summary = summary[summary["run_group_id"] == latest["run_group_id"]]
            chart_heading = f"Latest Run -- {latest['run_timestamp'] or 'unknown time'}"
            chart_subtitle = (
                f"dataset: {latest['dataset'] or 'n/a'}  |  model(s): {latest['models'] or 'n/a'}  |  "
                f"defense(s): {latest['defenses'] or 'n/a'}  |  {latest['rows']} rows"
            )
        else:
            latest_summary = summary
            chart_heading = "Latest Run"
            chart_subtitle = "No per-run tracking available for this data -- showing everything currently selected."

        if has_run_tracking:
            accordion_sections = [_run_history_header_row()] + [
                _run_accordion_section(
                    run_row, summary[summary["run_group_id"] == run_row["run_group_id"]], is_open=(i == 0)
                )
                for i, (_, run_row) in enumerate(run_history.iterrows())
            ]
        else:
            accordion_sections = [
                html.Div("No individual runs to show yet -- results predate per-run tracking.", className="muted")
            ]

        content = html.Div(
            [
                html.H3(chart_heading, style={"marginTop": 0, "marginBottom": "2px"}),
                html.Div(chart_subtitle, className="muted", style={"marginBottom": "10px"}),
                _leaderboard_charts(latest_summary),
                html.H3("Run History", style={"marginTop": "26px", "marginBottom": "2px"}),
                html.Div(
                    "Every experiment run currently selected by the sidebar filters -- grouped by when the run "
                    "was submitted (so every model/defense combination launched together stays together) and "
                    "sorted newest first. Expand a run to see its own model/defense breakdown.",
                    className="muted",
                    style={"marginBottom": "10px"},
                ),
                html.Div(accordion_sections),
            ],
            style={"marginTop": "16px"},
        )
    elif tab == "scenarios":
        content = html.Div(
            [
                html.H3("Attack Success by Scenario Category", style={"marginTop": 0, "marginBottom": "2px"}),
                html.Div(
                    "How often the injected attack succeeded in each scenario category, broken down by model "
                    "(attacked-condition rows only, across whatever the sidebar filters currently select). "
                    "Lower is better.",
                    className="muted",
                    style={"marginBottom": "10px"},
                ),
                dcc.Graph(figure=category_attack_rate(summary), config={"displaylogo": False}),
                html.H3("Scenario Vulnerability Heatmap", style={"marginTop": "20px", "marginBottom": "2px"}),
                html.Div(
                    "Attack success rate for every individual scenario, by model. Darker red means that model "
                    "was successfully attacked more often on that specific scenario -- useful for spotting "
                    "individual weak points rather than category-level trends.",
                    className="muted",
                    style={"marginBottom": "10px"},
                ),
                dcc.Graph(figure=task_heatmap(summary), config={"displaylogo": False}),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "8px", "marginTop": "16px"},
        )
    else:
        table_cols = [
            "run_timestamp", "model", "dataset", "scenario_id", "category", "attack_technique", "severity", "user_task",
            "defense", "condition", "task_success", "attack_success", "robust_success",
            "latency_seconds", "total_cost_usd", "error", "model_output",
        ]
        visible = summary[[c for c in table_cols if c in summary.columns]].head(300)
        content = html.Div(
            [
                html.H3("Run Drilldown", style={"marginTop": 0, "marginBottom": "2px"}),
                html.Div(
                    f"Every individual scenario row currently selected by the sidebar filters (showing up to "
                    f"300 of {len(summary)} rows). Sort or filter any column to dig into specific models, "
                    f"categories, or attack techniques.",
                    className="muted",
                    style={"marginBottom": "2px"},
                ),
                html.Div(
                    "Red row = the injected attack succeeded (defense failed). Green row = task success "
                    "under attack (task completed and the attack was resisted).",
                    className="muted",
                    style={"marginBottom": "10px"},
                ),
                dash_table.DataTable(
                    data=visible.to_dict("records"),
                    columns=[{"name": col, "id": col} for col in visible.columns],
                    page_size=12,
                    filter_action="native",
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#eef2ff", "fontWeight": "800", "border": "0"},
                    style_cell={"textAlign": "left", "whiteSpace": "normal", "height": "auto", "fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "12px", "padding": "10px", "border": f"1px solid {COLORS['line']}"},
                    style_data_conditional=[
                        {"if": {"filter_query": "{attack_success} = true"}, "backgroundColor": "#fef2f2"},
                        {"if": {"filter_query": "{robust_success} = true"}, "backgroundColor": "#ecfdf5"},
                    ],
                ),
            ],
            style={"marginTop": "16px"},
        )

    return (
        card("Evaluations Completed", str(len(summary)), f"{len(attacked)} attacked rows", COLORS["blue"]),
        card("Task Success Under Attack", pct(robust), "Task succeeds and attack fails", COLORS["green"]),
        card("Attack Success", pct(attack), "Lower is better", COLORS["red"]),
        card("Utility Change", pct(utility_drop), "Benign minus attacked utility", COLORS["amber"]),
        card("Est. Cost", total_cost_display, "Combined target and judge model spend", COLORS["amber"]),
        run_status,
        content,
    )


_DEFAULT_PROGRESS_STEPS = [{"label": "Ready", "status": "done", "detail": "Choose settings and click Run Experiment."}]
_DEFAULT_LOG_TEXT = "Ready. Choose settings and click Run Experiment. The log updates when the run completes."


def _progress_step_row(step: dict):
    icon = {"done": "OK", "running": "...", "error": "X"}.get(step["status"], "OK")
    return html.Div(
        [
            html.Div(icon, className=f"progress-step-icon progress-step-icon--{step['status']}"),
            html.Div(
                [
                    html.Div(step["label"], className="progress-step-label"),
                    html.Div(step["detail"], className="progress-step-detail") if step.get("detail") else None,
                ]
            ),
        ],
        className="progress-step",
    )


def _render_status_tab(tab_value: str, progress_data, log_text):
    if tab_value == "logs":
        return html.Pre(log_text or _DEFAULT_LOG_TEXT, className="logs-box")
    steps = progress_data or _DEFAULT_PROGRESS_STEPS
    return html.Div([_progress_step_row(step) for step in steps])


def _section_body_style(is_open: bool) -> dict:
    return {"display": "block"} if is_open else {"display": "none"}


def _section_chevron_class(is_open: bool) -> str:
    return "config-section-chevron config-section-chevron--open" if is_open else "config-section-chevron"


def _section_summary_style(is_open: bool) -> dict:
    return {"display": "none"} if is_open else {"display": "block"}


def _config_section(number: int, title: str, children: list, default_open: bool = False):
    return html.Div(
        [
            html.Div(
                [
                    html.Span(str(number), className="config-section-number"),
                    html.Span(title),
                    html.Span(id=f"config-section-summary-{number}", className="config-section-summary-text", style=_section_summary_style(default_open)),
                    html.Span("›", id=f"config-section-chevron-{number}", className=_section_chevron_class(default_open)),
                ],
                id=f"config-section-header-{number}",
                n_clicks=0,
                className="config-section-summary",
            ),
            html.Div(children, id=f"config-section-body-{number}", className="config-section-body", style=_section_body_style(default_open)),
        ],
        className="config-section",
    )


def _run_summary_metric(label: str, value: str):
    return html.Div(
        [
            html.Div(label, className="muted", style={"marginTop": 0}),
            html.Div(value, style={"fontWeight": 850, "fontSize": "15px"}),
        ]
    )


def experiment_panel():
    judging_enabled = bool(JUDGING_CFG.get("enabled"))
    judge_status = (
        "Responses are scored by an LLM judge, so each scenario makes one call for the "
        "response and a second call for the judge's evaluation. Pick which model acts as "
        "judge below."
        if judging_enabled
        else "LLM judging is disabled in configs/benchmark.yaml, so responses are scored with "
        "keyword matching only -- the judge model picker below has no effect until judging is enabled."
    )
    # Always rendered (not conditional on judging_enabled) so the run-judge-model component
    # always exists in the layout -- the run_experiment callback reads it as a State on every
    # click regardless of whether judging happens to be enabled.
    judge_picker = [
        html.Label("Judge Model", className="filter-label"),
        dcc.Dropdown(
            id="run-judge-model",
            options=model_dropdown_options(),
            value=JUDGING_CFG.get("judge_model"),
            clearable=False,
        ),
        html.Div(id="judge-model-key-status", style={"marginTop": "6px"}),
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Configure Experiment", style={"marginTop": 0}),
                            dcc.Store(id="open-config-section-store", data=1),
                            _config_section(
                                1,
                                "Dataset",
                                [
                                    html.Label("Dataset Source", className="filter-label"),
                                    dcc.RadioItems(
                                        id="run-dataset-source",
                                        options=[
                                            {
                                                "label": f"Default generated workplace dataset ({DEFAULT_SCENARIO_LIMIT_MAX} scenarios)",
                                                "value": "default",
                                            },
                                            {"label": "Uploaded JSONL dataset", "value": "uploaded"},
                                        ],
                                        value="default",
                                    ),
                                    html.Div(
                                        "Workplace tasks (Slack summaries, action items, incident reports, and more) "
                                        "each paired with a prompt-injection attack technique, used to test whether a "
                                        "model completes the real task without complying with the injected instruction.",
                                        className="muted",
                                        style={"marginTop": "4px"},
                                    ),
                                    html.Details(
                                        [
                                            html.Summary(
                                                "Dataset format for uploads",
                                                style={"cursor": "pointer", "fontWeight": 700, "fontSize": "13px", "color": COLORS["blue"], "marginTop": "10px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        ".jsonl (one JSON object per line), .json (one top-level JSON array "
                                                        "of objects), or .csv are all accepted. Required fields:"
                                                    ),
                                                    html.Ul([html.Li(html.Code(field)) for field in REQUIRED_SCENARIO_FIELDS]),
                                                    html.Div("Recommended fields (improve keyword/judge scoring quality if present):"),
                                                    html.Ul([html.Li(html.Code(field)) for field in RECOMMENDED_SCENARIO_FIELDS]),
                                                    html.Div("Example row (JSONL/JSON):", style={"marginTop": "4px"}),
                                                    html.Pre(
                                                        json.dumps(EXAMPLE_SCENARIO_ROW, indent=2),
                                                        style={
                                                            "background": "#f8fafc",
                                                            "border": f"1px solid {COLORS['line']}",
                                                            "borderRadius": "6px",
                                                            "padding": "10px",
                                                            "fontSize": "11px",
                                                            "overflowX": "auto",
                                                            "whiteSpace": "pre-wrap",
                                                        },
                                                    ),
                                                    html.Div(
                                                        "For CSV, use the same column names, with list-valued fields "
                                                        "(expected_keywords, forbidden_keywords) written as semicolon-separated "
                                                        "text in a single cell, e.g.:",
                                                        style={"marginTop": "8px"},
                                                    ),
                                                    html.Pre(
                                                        _example_csv_snippet(),
                                                        style={
                                                            "background": "#f8fafc",
                                                            "border": f"1px solid {COLORS['line']}",
                                                            "borderRadius": "6px",
                                                            "padding": "10px",
                                                            "fontSize": "11px",
                                                            "overflowX": "auto",
                                                            "whiteSpace": "pre",
                                                        },
                                                    ),
                                                    html.Div(
                                                        "An upload that's missing required fields, has duplicate scenario_ids, "
                                                        "or fails to parse will be rejected with a specific error before it's used.",
                                                        style={"marginTop": "4px"},
                                                    ),
                                                ],
                                                style={"fontSize": "13px", "color": COLORS["muted"], "marginTop": "6px"},
                                            ),
                                        ],
                                        style={"marginTop": "4px"},
                                    ),
                                    dcc.Upload(
                                        id="dataset-upload",
                                        children=html.Div(["Drag and drop or click to upload a .jsonl, .json, or .csv dataset"]),
                                        style={"border": f"1px dashed {COLORS['line']}", "borderRadius": "8px", "padding": "20px", "textAlign": "center", "marginTop": "12px", "background": "#f8fafc"},
                                        multiple=False,
                                        accept=".jsonl,.json,.csv",
                                    ),
                                    html.Div(id="upload-status", className="muted"),
                                ],
                                default_open=True,
                            ),
                            _config_section(
                                2,
                                "Models and Defenses",
                                [
                                    html.Div(judge_status, className="notice"),
                                    *judge_picker,
                                    html.Label("Models", className="filter-label"),
                                    dcc.Dropdown(id="run-models", options=model_dropdown_options(), value=["gpt-4o-mini"], multi=True),
                                    html.Div(id="model-key-status", style={"marginTop": "6px"}),
                                    html.Label("Defenses", className="filter-label"),
                                    dcc.Dropdown(id="run-defenses", options=defense_dropdown_options(), value=["baseline", "instruction_boundary"], multi=True),
                                    html.Div(id="defense-description", style={"marginTop": "6px"}),
                                ],
                            ),
                            _config_section(
                                3,
                                "Scenario Filters",
                                [
                                    html.Div(
                                        "Narrow which scenarios to run by whichever matters most to you -- scenario "
                                        "type (the underlying user task), attack technique, severity, or any combination. "
                                        "Leave any of these empty to include everything on that axis.",
                                        className="muted",
                                        style={"marginBottom": "4px"},
                                    ),
                                    html.Label("Scenario Type (user_task)", className="filter-label"),
                                    dcc.Dropdown(
                                        id="run-categories",
                                        options=[],
                                        value=[],
                                        multi=True,
                                        placeholder="All scenario types (default)",
                                    ),
                                    html.Label("Attack Technique", className="filter-label"),
                                    dcc.Dropdown(
                                        id="run-attack-techniques",
                                        options=[],
                                        value=[],
                                        multi=True,
                                        placeholder="All attack techniques (default)",
                                    ),
                                    html.Label("Severity", className="filter-label"),
                                    dcc.Dropdown(
                                        id="run-severities",
                                        options=[],
                                        value=[],
                                        multi=True,
                                        placeholder="All severities (default)",
                                    ),
                                    html.Div(
                                        "Severity reflects how blatant the injected ask is: low techniques bury or "
                                        "soften the ask, high techniques lean on urgency or authority.",
                                        className="muted",
                                        style={"marginTop": "4px"},
                                    ),
                                    html.Label("Number of Scenarios to Test", className="filter-label"),
                                    html.Div(
                                        "Required. The app randomly picks this many scenarios from whatever matches "
                                        "the filters above -- if you didn't set any filters, it samples randomly from "
                                        "the full dataset.",
                                        className="muted",
                                        style={"marginBottom": "4px"},
                                    ),
                                    dcc.Input(
                                        id="run-limit",
                                        type="number",
                                        min=1,
                                        max=DEFAULT_SCENARIO_LIMIT_MAX,
                                        step=1,
                                        value=min(10, DEFAULT_SCENARIO_LIMIT_MAX),
                                        style={"width": "120px"},
                                    ),
                                ],
                            ),
                            html.H4("Run Summary", style={"marginTop": "16px", "marginBottom": "2px"}),
                            html.Div(
                                "Estimated from average token counts in past runs -- actual cost may vary.",
                                className="muted",
                                style={"marginBottom": "6px"},
                            ),
                            html.Div(id="run-summary", className="notice"),
                            html.Div(
                                [
                                    html.Button("Run Experiment", id="run-button", n_clicks=0, className="button-primary"),
                                    html.Button("Refresh Results", id="refresh-button", n_clicks=0, className="button-secondary"),
                                ],
                                style={"marginTop": "18px"},
                            ),
                            html.Div("Runs can take a while because every selected model/defense/scenario makes real API calls.", className="muted"),
                        ],
                        className="card",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Experiment Status", style={"marginTop": 0, "marginBottom": 0}),
                                    html.Div("Idle", id="run-status-badge", className="status-badge status-badge--idle"),
                                ],
                                style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"},
                            ),
                            html.Div(
                                "Choose settings and click Run Experiment to see live progress here.",
                                id="status-idle-note",
                                className="muted",
                                style={"marginTop": "10px"},
                            ),
                            html.Div(
                                [
                                    dcc.Tabs(
                                        id="status-tabs",
                                        value="progress",
                                        className="subtab-container",
                                        children=[
                                            dcc.Tab(label="Progress", value="progress", className="subtab", selected_className="subtab--selected"),
                                            dcc.Tab(label="Logs", value="logs", className="subtab", selected_className="subtab--selected"),
                                        ],
                                    ),
                                    dcc.Store(id="run-progress-data"),
                                    dcc.Store(id="run-log-data"),
                                    dcc.Loading(html.Div(id="status-tab-content", style={"marginTop": "12px"}), type="circle"),
                                ],
                                id="status-body-wrapper",
                                style={"display": "none"},
                            ),
                        ],
                        className="card",
                    ),
                ],
                style={"display": "grid", "gridTemplateColumns": "minmax(380px, 1fr) minmax(380px, 1fr)", "gap": "16px"},
            ),
        ],
        style={"marginTop": "16px"},
    )


@app.callback(Output("upload-status", "children"), Input("dataset-upload", "filename"))
def show_upload_name(filename):
    if not filename:
        return "Upload is optional. The generated dataset is selected by default."
    return f"Selected file: {filename}"


@app.callback(
    Output("status-tab-content", "children"),
    Input("status-tabs", "value"),
    Input("run-progress-data", "data"),
    Input("run-log-data", "data"),
)
def render_status_tab(tab_value, progress_data, log_text):
    return _render_status_tab(tab_value, progress_data, log_text)


@app.callback(
    Output("run-summary", "children"),
    Input("run-models", "value"),
    Input("run-defenses", "value"),
    Input("run-limit", "value"),
    Input("run-judge-model", "value"),
)
def update_run_summary(models, defenses, scenario_count, judge_model):
    models = models or []
    defenses = defenses or []
    scenario_count = int(scenario_count) if scenario_count else 0
    judging_enabled = bool(JUDGING_CFG.get("enabled"))

    rows_per_model = scenario_count * len(defenses) * 2  # one row each for benign and attacked
    total_evaluations = rows_per_model * len(models)

    pricing_known = True
    target_cost = 0.0
    for model in models:
        pricing = get_model_pricing(_PROJECT_CONFIG, model)
        if pricing is None:
            pricing_known = False
            continue
        target_cost += rows_per_model * compute_cost_usd(pricing, _AVG_PROMPT_TOKENS, _AVG_COMPLETION_TOKENS)

    api_calls = total_evaluations
    judge_cost = 0.0
    if judging_enabled and judge_model:
        api_calls += total_evaluations
        judge_pricing = get_model_pricing(_PROJECT_CONFIG, judge_model)
        if judge_pricing is None:
            pricing_known = False
        else:
            judge_cost = total_evaluations * compute_cost_usd(judge_pricing, _AVG_PROMPT_TOKENS, _AVG_COMPLETION_TOKENS)

    total_cost = target_cost + judge_cost
    cost_display = f"~${total_cost:.4f}" + ("" if pricing_known else " (some pricing unknown)")

    return html.Div(
        [
            _run_summary_metric("Scenarios", str(scenario_count)),
            _run_summary_metric("Models", str(len(models))),
            _run_summary_metric("Defenses", str(len(defenses))),
            _run_summary_metric("Total Evaluations", str(total_evaluations)),
            _run_summary_metric("Est. API Calls", str(api_calls)),
            _run_summary_metric("Est. Cost", cost_display),
        ],
        className="run-summary-grid",
    )


_CONFIG_SECTION_HEADER_TO_NUMBER = {
    "config-section-header-1": 1,
    "config-section-header-2": 2,
    "config-section-header-3": 3,
}


@app.callback(
    Output("config-section-body-1", "style"),
    Output("config-section-body-2", "style"),
    Output("config-section-body-3", "style"),
    Output("config-section-chevron-1", "className"),
    Output("config-section-chevron-2", "className"),
    Output("config-section-chevron-3", "className"),
    Output("config-section-summary-1", "style"),
    Output("config-section-summary-2", "style"),
    Output("config-section-summary-3", "style"),
    Output("open-config-section-store", "data"),
    Input("config-section-header-1", "n_clicks"),
    Input("config-section-header-2", "n_clicks"),
    Input("config-section-header-3", "n_clicks"),
    State("open-config-section-store", "data"),
)
def toggle_config_section(_c1, _c2, _c3, currently_open):
    clicked_number = _CONFIG_SECTION_HEADER_TO_NUMBER.get(ctx.triggered_id)
    if clicked_number is None:
        open_section = currently_open
    elif clicked_number == currently_open:
        open_section = None  # clicking the already-open section's header collapses it
    else:
        open_section = clicked_number

    is_open = {n: (n == open_section) for n in (1, 2, 3)}
    return (
        _section_body_style(is_open[1]),
        _section_body_style(is_open[2]),
        _section_body_style(is_open[3]),
        _section_chevron_class(is_open[1]),
        _section_chevron_class(is_open[2]),
        _section_chevron_class(is_open[3]),
        _section_summary_style(is_open[1]),
        _section_summary_style(is_open[2]),
        _section_summary_style(is_open[3]),
        open_section,
    )


def _summarize_list(values: list[str], all_label: str, limit: int = 2) -> str:
    if not values:
        return all_label
    if len(values) <= limit:
        return ", ".join(values)
    return ", ".join(values[:limit]) + f" +{len(values) - limit} more"


@app.callback(
    Output("config-section-summary-1", "children"),
    Input("run-dataset-source", "value"),
    Input("upload-status", "children"),
)
def summarize_dataset_section(dataset_source, upload_status):
    if dataset_source == "uploaded":
        return f"Uploaded -- {upload_status}" if upload_status else "Uploaded dataset"
    return f"Default generated dataset ({DEFAULT_SCENARIO_LIMIT_MAX} scenarios)"


@app.callback(
    Output("config-section-summary-2", "children"),
    Input("run-models", "value"),
    Input("run-defenses", "value"),
    Input("run-judge-model", "value"),
)
def summarize_models_section(models, defenses, judge_model):
    models = models or []
    defenses = defenses or []
    parts = [
        f"Models: {_summarize_list(models, 'none selected')}",
        f"Defenses: {_summarize_list(defenses, 'none selected')}",
    ]
    if bool(JUDGING_CFG.get("enabled")) and judge_model:
        parts.append(f"Judge: {judge_model}")
    return "  •  ".join(parts)


@app.callback(
    Output("config-section-summary-3", "children"),
    Input("run-categories", "value"),
    Input("run-attack-techniques", "value"),
    Input("run-severities", "value"),
    Input("run-limit", "value"),
)
def summarize_filters_section(categories, attack_techniques, severities, scenario_count):
    categories = categories or []
    attack_techniques = attack_techniques or []
    severities = severities or []
    scenario_count = int(scenario_count) if scenario_count else 0
    parts = [
        _summarize_list(categories, "all types"),
        _summarize_list(attack_techniques, "all techniques"),
        _summarize_list(severities, "all severities"),
        f"{scenario_count} scenario{'s' if scenario_count != 1 else ''}",
    ]
    return "  •  ".join(parts)


@app.callback(Output("model-key-status", "children"), Input("run-models", "value"))
def show_model_key_status(selected_models):
    if not selected_models:
        return ""
    missing = missing_api_key_models(_PROJECT_CONFIG, selected_models)
    if not missing:
        return ""
    lines = [
        f"No API key for {model_id} -- this model needs {env_var} ({_provider_label(provider_id)}) set in .env."
        for model_id, provider_id, env_var in missing
    ]
    return html.Div(
        [html.Div(line) for line in lines],
        className="notice",
        style={"borderColor": COLORS["red"], "color": COLORS["red"], "marginTop": "4px"},
    )


@app.callback(Output("judge-model-key-status", "children"), Input("run-judge-model", "value"))
def show_judge_model_key_status(judge_model):
    if not judge_model:
        return ""
    missing = missing_api_key_models(_PROJECT_CONFIG, [judge_model])
    if not missing:
        return ""
    model_id, provider_id, env_var = missing[0]
    return html.Div(
        f"No API key for {model_id} -- this judge model needs {env_var} ({_provider_label(provider_id)}) set in .env.",
        className="notice",
        style={"borderColor": COLORS["red"], "color": COLORS["red"], "marginTop": "4px"},
    )


@app.callback(Output("defense-description", "children"), Input("run-defenses", "value"))
def show_defense_descriptions(selected_defenses):
    if not selected_defenses:
        return ""
    return html.Div(
        [
            html.Div([html.B(f"{get_defense_info(defense_id)['label']}: "), get_defense_info(defense_id)["description"]])
            for defense_id in selected_defenses
        ],
        className="notice",
        style={"marginTop": "4px"},
    )


def _preview_scenario_path(dataset_source: str) -> Path:
    if dataset_source == "uploaded":
        return UPLOAD_PATH
    return resolve_path(_PROJECT_CONFIG.benchmark["datasets"]["custom"]["path"])


def _load_preview_scenarios(dataset_source: str) -> list[dict] | None:
    path = _preview_scenario_path(dataset_source)
    if not path.exists():
        return None
    try:
        return read_jsonl(path)
    except json.JSONDecodeError:
        return None


@app.callback(
    Output("run-categories", "options"),
    Output("run-attack-techniques", "options"),
    Output("run-severities", "options"),
    Input("run-dataset-source", "value"),
    Input("upload-status", "children"),  # refreshes options once an upload finishes
)
def update_scenario_filter_options(dataset_source, _upload_status):
    scenarios = _load_preview_scenarios(dataset_source) or []
    return (
        category_dropdown_options(scenarios),
        attack_technique_dropdown_options(scenarios),
        severity_dropdown_options(scenarios),
    )


@app.callback(
    Output("run-limit", "max"),
    Output("run-limit", "value"),
    Output("run-limit", "disabled"),
    Input("run-dataset-source", "value"),
    Input("upload-status", "children"),
    Input("run-categories", "value"),
    Input("run-attack-techniques", "value"),
    Input("run-severities", "value"),
    State("run-limit", "value"),
)
def update_run_limit_bounds(dataset_source, _upload_status, categories, attack_techniques, severities, current_value):
    scenarios = _load_preview_scenarios(dataset_source) or []
    matched = filter_scenario_rows(
        scenarios, categories=categories, attack_techniques=attack_techniques, severities=severities
    )
    total = len(matched)

    if total == 0:
        return 1, current_value, True

    new_value = min(int(current_value), total) if current_value else min(10, total)
    return total, new_value, False


@app.callback(
    Output("run-log-data", "data"),
    Output("run-progress-data", "data"),
    Output("run-status-badge", "children"),
    Output("run-status-badge", "className"),
    Output("run-completed-store", "data"),
    Output("status-idle-note", "style"),
    Output("status-body-wrapper", "style"),
    Input("run-button", "n_clicks"),
    Input("refresh-button", "n_clicks"),
    State("run-dataset-source", "value"),
    State("dataset-upload", "contents"),
    State("dataset-upload", "filename"),
    State("run-models", "value"),
    State("run-defenses", "value"),
    State("run-categories", "value"),
    State("run-attack-techniques", "value"),
    State("run-severities", "value"),
    State("run-limit", "value"),
    State("run-judge-model", "value"),
    prevent_initial_call=True,
    running=[
        (Output("run-button", "disabled"), True, False),
        (Output("refresh-button", "disabled"), True, False),
        (
            Output("kpi-running-banner", "children"),
            "Running experiment now -- the KPI cards and charts below still show the previous run's "
            "results until this finishes.",
            "",
        ),
        (Output("kpi-running-banner", "style"), {"display": "block", "marginBottom": "10px"}, {"display": "none"}),
        (Output("run-status-badge", "children"), "Running...", no_update),
        (Output("run-status-badge", "className"), "status-badge status-badge--running", no_update),
        # Expand the Experiment Status panel for the duration of the run -- it starts compact
        # (idle note only) and stays expanded once a real run/refresh has actually happened.
        # The "after" value must be an explicit dict, not no_update: for a dict-shaped prop like
        # style, Dash resolves a no_update resting value to an empty object at initial page load
        # (before the callback has ever fired), which would make the panel render expanded even
        # while idle. Using the literal idle-state dict here keeps it correct from first load,
        # and the final return statements below override it once a real run actually finishes.
        (Output("status-idle-note", "style"), {"display": "none"}, {"display": "block", "marginTop": "10px"}),
        (Output("status-body-wrapper", "style"), {"display": "block"}, {"display": "none"}),
    ],
)
def run_experiment(
    run_clicks, refresh_clicks, dataset_source, upload_contents, upload_filename, models, defenses,
    categories, attack_techniques, severities, overall_cap, judge_model,
):
    global _LAST_HANDLED_RUN_CLICKS
    trigger = ctx.triggered_id
    logs: list[str] = []
    progress: list[dict] = []

    def add_step(label, detail=None, status="done"):
        progress.append({"label": label, "status": status, "detail": detail})

    try:
        if trigger == "run-button":
            # Defense-in-depth against Dash's remount quirk (an Input-bound component re-fires its
            # callback if it gets unmounted/remounted elsewhere on the page, regardless of
            # prevent_initial_call): only ever launch a real run for a click count strictly higher
            # than the last one we actually handled, never for the same or a reset-to-default value.
            if run_clicks is None or run_clicks <= _LAST_HANDLED_RUN_CLICKS:
                return no_update, no_update, no_update, no_update, no_update, no_update, no_update
            _LAST_HANDLED_RUN_CLICKS = run_clicks

            if not models:
                raise ValueError("Select at least one model.")
            if not defenses:
                raise ValueError("Select at least one defense.")
            if not overall_cap or int(overall_cap) < 1:
                raise ValueError("Enter the number of scenarios to test (Step 3).")

            missing = missing_api_key_models(_PROJECT_CONFIG, models)
            if missing:
                lines = [
                    f"  - {model_id}: provider \"{_provider_label(provider_id)}\" requires {env_var}"
                    for model_id, provider_id, env_var in missing
                ]
                raise ValueError(
                    "Cannot run: the API key is missing for one or more selected models.\n"
                    + "\n".join(lines)
                    + "\n\nAdd the missing key(s) to .env and restart the app."
                )

            judging_enabled = bool(JUDGING_CFG.get("enabled"))
            if judging_enabled:
                if not judge_model:
                    raise ValueError("Select a judge model (LLM judging is enabled).")
                judge_missing = missing_api_key_models(_PROJECT_CONFIG, [judge_model])
                if judge_missing:
                    model_id, provider_id, env_var = judge_missing[0]
                    raise ValueError(
                        f"Cannot run: the judge model {model_id} requires {env_var} "
                        f"(provider \"{_provider_label(provider_id)}\"), which is missing from .env."
                    )

            add_step("Validate settings")

            dataset_label = "custom"
            if dataset_source == "uploaded":
                count = save_uploaded_dataset(upload_contents, upload_filename)
                base_scenarios = read_jsonl(UPLOAD_PATH)
                dataset_label = "uploaded"
                logs.append(f"Saved uploaded dataset ({upload_filename}) with {count} scenario rows.")
            else:
                base_scenarios = read_jsonl(resolve_path(_PROJECT_CONFIG.benchmark["datasets"]["custom"]["path"]))

            matched_scenarios = filter_scenario_rows(
                base_scenarios, categories=categories, attack_techniques=attack_techniques, severities=severities
            )
            if not matched_scenarios:
                raise ValueError("No scenarios match your scenario type / attack technique / severity selection.")

            requested = int(overall_cap)
            if requested >= len(matched_scenarios):
                selected_scenarios = matched_scenarios
                if requested > len(matched_scenarios):
                    logs.append(
                        f"Only {len(matched_scenarios)} scenarios matched your filters -- requested {requested}, "
                        f"using all {len(matched_scenarios)}."
                    )
            else:
                selected_scenarios = random.sample(matched_scenarios, requested)

            write_scenario_rows_jsonl(selected_scenarios, SELECTED_PATH)
            select_detail = (
                f"Selected {len(selected_scenarios)} of {len(base_scenarios)} scenarios "
                f"(across {len(category_counts(selected_scenarios))} scenario type(s))."
            )
            logs.append(select_detail)
            add_step("Select scenarios", detail=select_detail)
            scenario_arg = ["--scenario-path", str(SELECTED_PATH)]

            # One run_group_id shared by every model/defense combination launched from this
            # single click, so they're grouped together as one run in the Overview tab
            # (by when the experiment was submitted) instead of one run per subprocess call.
            run_group_id = str(int(time.time()))
            logs.append("Starting experiment...")
            for model in models:
                for defense in defenses:
                    command = [
                        "scripts/run_benchmark.py",
                        "--dataset",
                        "custom",
                        "--mode",
                        "full",
                        "--model",
                        model,
                        "--defense",
                        defense,
                        "--dataset-label",
                        dataset_label,
                        "--run-group-id",
                        run_group_id,
                    ]
                    command.extend(scenario_arg)
                    if judging_enabled and judge_model:
                        command.extend(["--judge-model", judge_model])
                    logs.append(f"Running {model} / {defense}...")
                    logs.append(run_command(command))
                    add_step(f"Run {model} / {defense}")

        logs.append("Refreshing processed analytics...")
        logs.append(run_command(["scripts/parse_results.py"]))
        logs.append(run_command(["scripts/build_analytics_db.py"]))
        logs.append(run_command(["scripts/make_report.py"]))
        add_step("Refresh analytics")
        complete_detail = "Open Overview, Scenario Analysis, or Run Drilldown to inspect the updated results."
        logs.append(f"Complete. {complete_detail}")
        add_step("Complete", detail=complete_detail)
        return (
            "\n\n".join(logs),
            progress,
            "Complete",
            "status-badge status-badge--done",
            str(time.time()),
            {"display": "none"},
            {"display": "block"},
        )
    except Exception as exc:
        logs.append(f"Experiment failed:\n{exc}")
        add_step("Failed", detail=str(exc), status="error")
        return (
            "\n\n".join(logs),
            progress,
            "Failed",
            "status-badge status-badge--error",
            no_update,
            {"display": "none"},
            {"display": "block"},
        )


if __name__ == "__main__":
    app.run(debug=False)
