from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


TEMPLATE = "plotly_white"
COLORWAY = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c"]


def polish(fig):
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
    return fig


def leaderboard_bar(leaderboard: pd.DataFrame):
    if leaderboard.empty:
        return go.Figure()
    fig = px.bar(
        leaderboard,
        x="model",
        y="robust_success_rate",
        color="defense",
        barmode="group",
        title="Task Success Under Attack Rate by Model and Defense",
        labels={"robust_success_rate": "Task success under attack rate"},
    )
    return polish(fig)


def utility_vs_attack(leaderboard: pd.DataFrame):
    if leaderboard.empty:
        return go.Figure()
    fig = px.scatter(
        leaderboard,
        x="attack_success_rate",
        y="attacked_success",
        color="model",
        symbol="defense",
        size="runs",
        title="Utility vs Attack Success",
        labels={
            "attack_success_rate": "Attack success rate",
            "attacked_success": "Task success under attack",
        },
    )
    return polish(fig)


def category_attack_rate(summary: pd.DataFrame):
    attacked = summary[summary["condition"] == "attacked"]
    if attacked.empty:
        return go.Figure()
    grouped = (
        attacked.groupby(["category", "model"], dropna=False)["attack_success"]
        .mean()
        .reset_index()
    )
    fig = px.bar(
        grouped,
        x="category",
        y="attack_success",
        color="model",
        barmode="group",
        title="Attack Success by Scenario Category",
    )
    return polish(fig)


def task_heatmap(summary: pd.DataFrame):
    attacked = summary[summary["condition"] == "attacked"]
    if attacked.empty:
        return go.Figure()
    pivot = attacked.pivot_table(
        index="scenario_id",
        columns="model",
        values="attack_success",
        aggfunc="mean",
        fill_value=0,
    )
    fig = px.imshow(
        pivot,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Reds",
        title="Scenario Vulnerability Heatmap",
    )
    return polish(fig)
