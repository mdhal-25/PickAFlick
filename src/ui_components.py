"""
Reusable Streamlit UI pieces: cards, explanations, loading helpers.
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from src import api


def movie_card(movie: dict[str, Any], *, key_prefix: str = "m") -> None:
    """Compact tile: poster, title, rating, year."""
    mid = movie.get("id", 0)
    title = movie.get("title") or "Untitled"
    poster = api.poster_url(movie.get("poster_path"))
    rating = movie.get("vote_average")
    rd = movie.get("release_date") or ""
    year = rd[:4] if len(rd) >= 4 else "—"

    with st.container():
        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.caption("*No poster*")
        st.markdown(f"**{title}**  \n`{year}`  \n⭐ {rating if rating is not None else '—'}")


def movie_grid(
    movies: list[dict[str, Any]],
    *,
    cols: int = 4,
    key_prefix: str = "grid",
) -> None:
    """Render movies in a responsive column grid."""
    if not movies:
        st.info("No movies to show.")
        return
    for row_start in range(0, len(movies), cols):
        row = movies[row_start : row_start + cols]
        columns = st.columns(cols)
        for col, m in zip(columns, row):
            with col:
                movie_card(m, key_prefix=f"{key_prefix}_{m.get('id')}")


def explanation_block(text: str) -> None:
    st.markdown(f"> {text}")


def with_spinner(label: str, fn: Callable[[], Any]) -> Any:
    """Run a callable inside a standardized Streamlit spinner."""
    with st.spinner(label):
        return fn()


def error_bubble(msg: str) -> None:
    st.error(msg)


def warning_bubble(msg: str) -> None:
    st.warning(msg)
