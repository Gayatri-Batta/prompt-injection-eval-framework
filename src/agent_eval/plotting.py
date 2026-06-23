from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


TEMPLATE = "plotly_white"
COLORWAY = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c"]

# How many of the worst-performing scenarios to show in the vulnerability ranking --
# bounds the chart's height regardless of how large the underlying dataset grows.
TOP_N_WORST_SCENARIOS = 15


def polish(fig, percent_axis: str | None = None):
    fig.update_layout(
        template=TEMPLATE,
        colorway=COLORWAY,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=24, t=56, b=40),
        font=dict(family="Segoe UI, Arial, sans-serif", color="#172033"),
        title=dict(font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#e5e7eb", zeroline=False)
    if percent_axis in ("x", "both"):
        fig.update_xaxes(tickformat=".0%")
    if percent_axis in ("y", "both"):
        fig.update_yaxes(tickformat=".0%")
    return fig


def leaderboard_bar(leaderboard: pd.DataFrame):
    if leaderboard.empty:
        return go.Figure()
    # leaderboard is already sorted best-first by aggregate_leaderboard(); the first row for
    # each model is that model's best-performing defense, so this order reads best-to-worst.
    model_order = leaderboard.drop_duplicates("model")["model"].tolist()
    fig = px.bar(
        leaderboard,
        x="model",
        y="robust_success_rate",
        color="defense",
        barmode="group",
        text_auto=".0%",
        category_orders={"model": model_order},
        title="Which model/defense combo resists attacks best?",
        labels={"robust_success_rate": "Task success under attack rate", "model": "Model", "defense": "Defense"},
    )
    fig.update_traces(textposition="outside")
    mean_rate = leaderboard["robust_success_rate"].mean()
    fig.add_hline(
        y=mean_rate,
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text="avg",
        annotation_position="top left",
        annotation_font_color="#64748b",
    )
    return polish(fig, percent_axis="y")


def utility_vs_attack(leaderboard: pd.DataFrame):
    if leaderboard.empty:
        return go.Figure()
    plot_df = leaderboard.copy()
    plot_df["model_defense"] = plot_df["model"] + " / " + plot_df["defense"]
    fig = px.scatter(
        plot_df,
        x="attack_success_rate",
        y="attacked_success",
        color="model_defense",
        text="model_defense",
        hover_data=["runs"],
        title="Which combos stay both safe and useful?",
        labels={
            "attack_success_rate": "Attack success rate",
            "attacked_success": "Task success under attack",
            "model_defense": "Model / defense",
        },
    )
    fig.update_traces(textposition="top center", marker=dict(size=11))
    x_mid = plot_df["attack_success_rate"].median()
    y_mid = plot_df["attacked_success"].median()
    fig.add_vline(x=x_mid, line_dash="dot", line_color="#cbd5e1")
    fig.add_hline(y=y_mid, line_dash="dot", line_color="#cbd5e1")
    quadrant_labels = [
        ("Safe & useful", 0.02, 0.98),
        ("Unsafe & useful", 0.98, 0.98),
        ("Safe but low utility", 0.02, 0.02),
        ("Unsafe & low utility", 0.98, 0.02),
    ]
    for text, x, y in quadrant_labels:
        fig.add_annotation(
            text=text,
            xref="paper",
            yref="paper",
            x=x,
            y=y,
            showarrow=False,
            font=dict(size=11, color="#94a3b8"),
            xanchor="left" if x < 0.5 else "right",
            yanchor="bottom" if y < 0.5 else "top",
        )
    return polish(fig, percent_axis="both")


def category_attack_rate(summary: pd.DataFrame, show_sample_size: bool = False):
    attacked = summary[summary["condition"] == "attacked"]
    if attacked.empty:
        return go.Figure()
    grouped = (
        attacked.groupby(["category", "model"], dropna=False)["attack_success"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "attack_success", "size": "n"})
    )
    category_order = (
        attacked.groupby("category")["attack_success"].mean().sort_values(ascending=False).index.tolist()
    )
    bar_kwargs = dict(
        x="category",
        y="attack_success",
        color="model",
        barmode="group",
        category_orders={"category": category_order},
        title="Which scenario categories are easiest to attack?",
        labels={"attack_success": "Attack success rate", "category": "Scenario category"},
    )
    if show_sample_size:
        # Pooling across runs means a model/category bar may rest on very few (or zero) attempts --
        # show the sample size directly on the bar so a thin bar doesn't read as a confident result.
        grouped["bar_text"] = grouped.apply(lambda r: f"{r['attack_success'] * 100:.0f}% (n={int(r['n'])})", axis=1)
        fig = px.bar(grouped, text="bar_text", **bar_kwargs)
    else:
        fig = px.bar(grouped, text_auto=".0%", **bar_kwargs)
    fig.update_traces(textposition="outside")
    overall_rate = attacked["attack_success"].mean()
    fig.add_hline(
        y=overall_rate,
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text="avg",
        annotation_position="top left",
        annotation_font_color="#64748b",
    )
    return polish(fig, percent_axis="y")


def _readable(text) -> str:
    return str(text).replace("_", " ").strip().title()


def _scenario_labels(attacked: pd.DataFrame, scenario_ids: list[str]) -> dict[str, str]:
    """Human-readable label per scenario_id built from category + attack_technique. Only
    `category` is guaranteed present (required for any custom upload); `attack_technique` is
    merely recommended, and the two together are NOT guaranteed unique -- a dataset can easily
    have several scenarios sharing one category/technique pair. So the readable part is just
    context, and the actual scenario_id (required + validated-unique at upload time, see
    validate_scenario_rows in custom_dataset.py) is always appended -- guaranteed unique and
    traceable back to a row in Detailed Results, unlike a counter suffix that tells you nothing."""
    first_rows = attacked[attacked["scenario_id"].isin(scenario_ids)].drop_duplicates("scenario_id").set_index("scenario_id")
    labels: dict[str, str] = {}
    for sid in scenario_ids:
        row = first_rows.loc[sid] if sid in first_rows.index else None
        category = row.get("category") if row is not None else None
        technique = row.get("attack_technique") if row is not None else None
        if pd.notna(category) and pd.notna(technique):
            context = f"{_readable(category)} - {_readable(technique)}"
        elif pd.notna(category):
            context = _readable(category)
        else:
            context = None
        labels[sid] = f"{context} ({sid})" if context else str(sid)
    return labels


def task_heatmap(summary: pd.DataFrame, show_sample_size: bool = False):
    attacked = summary[summary["condition"] == "attacked"]
    if attacked.empty:
        return go.Figure()
    by_scenario = attacked.groupby("scenario_id")["attack_success"].mean().sort_values(ascending=False)
    worst_ids = by_scenario.head(TOP_N_WORST_SCENARIOS).index.tolist()
    labels = _scenario_labels(attacked, worst_ids)
    filtered = attacked[attacked["scenario_id"].isin(worst_ids)].copy()
    filtered["scenario_label"] = filtered["scenario_id"].map(labels)
    label_order = [labels[sid] for sid in worst_ids]
    grouped = (
        filtered.groupby(["scenario_label", "model"], dropna=False)["attack_success"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "attack_success", "size": "n"})
    )
    shown = len(worst_ids)
    bar_kwargs = dict(
        x="attack_success",
        y="scenario_label",
        color="model",
        orientation="h",
        barmode="group",
        category_orders={"scenario_label": label_order},
        title=f"Which scenarios are most vulnerable to attack? (top {shown})",
        labels={"attack_success": "Attack success rate", "scenario_label": "Scenario"},
    )
    if show_sample_size:
        grouped["bar_text"] = grouped.apply(lambda r: f"{r['attack_success'] * 100:.0f}% (n={int(r['n'])})", axis=1)
        fig = px.bar(grouped, text="bar_text", **bar_kwargs)
    else:
        fig = px.bar(grouped, text_auto=".0%", **bar_kwargs)
    fig.update_traces(textposition="outside")
    return polish(fig, percent_axis="x")
