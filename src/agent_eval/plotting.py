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


def category_attack_rate(summary: pd.DataFrame):
    attacked = summary[summary["condition"] == "attacked"]
    if attacked.empty:
        return go.Figure()
    grouped = (
        attacked.groupby(["category", "model"], dropna=False)["attack_success"]
        .mean()
        .reset_index()
    )
    category_order = (
        attacked.groupby("category")["attack_success"].mean().sort_values(ascending=False).index.tolist()
    )
    fig = px.bar(
        grouped,
        x="category",
        y="attack_success",
        color="model",
        barmode="group",
        text_auto=".0%",
        category_orders={"category": category_order},
        title="Which scenario categories are easiest to attack?",
        labels={"attack_success": "Attack success rate", "category": "Scenario category"},
    )
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


def task_heatmap(summary: pd.DataFrame):
    attacked = summary[summary["condition"] == "attacked"]
    if attacked.empty:
        return go.Figure()
    by_scenario = attacked.groupby("scenario_id")["attack_success"].mean().sort_values(ascending=False)
    worst_ids = by_scenario.head(TOP_N_WORST_SCENARIOS).index.tolist()
    filtered = attacked[attacked["scenario_id"].isin(worst_ids)].copy()
    filtered["scenario_label"] = filtered["scenario_id"].astype(str).str.replace("_", " ").str.title()
    label_order = [str(sid).replace("_", " ").title() for sid in worst_ids]
    grouped = filtered.groupby(["scenario_label", "model"], dropna=False)["attack_success"].mean().reset_index()
    shown = len(worst_ids)
    fig = px.bar(
        grouped,
        x="attack_success",
        y="scenario_label",
        color="model",
        orientation="h",
        barmode="group",
        text_auto=".0%",
        category_orders={"scenario_label": label_order},
        title=f"Which scenarios are most vulnerable to attack? (top {shown})",
        labels={"attack_success": "Attack success rate", "scenario_label": "Scenario"},
    )
    fig.update_traces(textposition="outside")
    return polish(fig, percent_axis="x")
