"""
Plotly charts for Insights tab.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _blank_figure(title: str, note: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_dark")
    if note:
        fig.add_annotation(
            text=note,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=13, color="#c7c7c7"),
        )
    return fig


def _movies_to_df(movies: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for m in movies:
        genres = [g.get("name") for g in (m.get("genres") or []) if isinstance(g, dict)]
        rd = m.get("release_date") or ""
        year = int(rd[:4]) if len(rd) >= 4 else None
        rows.append(
            {
                "title": m.get("title"),
                "vote_average": m.get("vote_average"),
                "vote_count": m.get("vote_count"),
                "runtime": m.get("runtime"),
                "genre": genres[0] if genres else "Unknown",
                "all_genres": ", ".join(genres) if genres else "",
                "year": year,
            }
        )
    # If rows is empty, DataFrame will have no columns; downstream charts must guard.
    return pd.DataFrame(rows)


def ratings_distribution_fig(movies: list[dict[str, Any]]) -> go.Figure:
    df = _movies_to_df(movies)
    if df is None or df.empty or "vote_average" not in df.columns:
        return _blank_figure("Ratings distribution unavailable", "No rating data to plot.")
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df = df.dropna(subset=["vote_average"])
    if df.empty:
        return _blank_figure("Ratings distribution unavailable", "No usable numeric ratings found.")
    fig = px.histogram(
        df,
        x="vote_average",
        nbins=20,
        color_discrete_sequence=["#7fdbff"],
    )
    fig.update_layout(
        template="plotly_dark",
        title="Ratings distribution (vote average)",
        xaxis_title="Vote average",
        yaxis_title="Count",
        bargap=0.15,
    )
    return fig


def genre_frequency_fig(movies: list[dict[str, Any]]) -> go.Figure:
    genre_counts: dict[str, int] = {}
    for m in movies:
        for g in m.get("genres") or []:
            if isinstance(g, dict) and g.get("name"):
                name = str(g["name"])
                genre_counts[name] = genre_counts.get(name, 0) + 1
    if not genre_counts:
        return _blank_figure("Genre frequency unavailable", "No genre data to plot.")
    df = pd.DataFrame(
        [{"genre": k, "count": v} for k, v in sorted(genre_counts.items(), key=lambda x: -x[1])]
    )
    fig = px.bar(df.head(18), x="genre", y="count", color="count", color_continuous_scale="Blues")
    fig.update_layout(
        template="plotly_dark",
        title="Genre frequency (top titles)",
        xaxis_title="Genre",
        yaxis_title="Count",
        coloraxis_showscale=False,
    )
    return fig


def runtime_distribution_fig(movies: list[dict[str, Any]]) -> go.Figure:
    df = _movies_to_df(movies)
    if df is None or df.empty or "runtime" not in df.columns:
        return _blank_figure("Runtime distribution unavailable", "Runtime data is missing for these titles.")

    # Coerce runtime safely to numeric minutes.
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df = df.dropna(subset=["runtime"])
    if df.empty:
        return _blank_figure("Runtime distribution unavailable", "No usable numeric runtime values found.")
    df = df[df["runtime"] > 0]
    if df.empty:
        return _blank_figure("Runtime distribution unavailable", "No positive runtime values found.")
    fig = px.histogram(
        df,
        x="runtime",
        nbins=25,
        color_discrete_sequence=["#ff6b6b"],
    )
    fig.update_layout(
        template="plotly_dark",
        title="Runtime distribution (minutes)",
        xaxis_title="Minutes",
        yaxis_title="Count",
        bargap=0.12,
    )
    return fig
