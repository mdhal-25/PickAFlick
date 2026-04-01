"""
PickAFlick — Streamlit movie recommender (TMDB + TF-IDF + sentiment + filters).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from src import api
from src import filters as filter_mod
from src import nlp_utils
from src import recommender
from src import ui_components
from src import visuals

load_dotenv()

QUICK_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Drama",
    "Family",
    "Fantasy",
    "Horror",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Thriller",
    "War",
]

ERA_MAP: dict[str, tuple[int, int] | None] = {
    "Any": None,
    "Classic (before 1980)": (1900, 1979),
    "80s": (1980, 1989),
    "90s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2029),
}

SHOW_DEBUG = False


def _ensure_watchlist_state() -> None:
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []  # list[dict] minimal movie records


def _watchlist_ids() -> set[int]:
    return {int(m["id"]) for m in st.session_state.watchlist if m.get("id") is not None}


def add_to_watchlist(movie: dict[str, Any]) -> None:
    _ensure_watchlist_state()
    mid = movie.get("id")
    if mid is None:
        return
    if int(mid) in _watchlist_ids():
        return
    st.session_state.watchlist.append(
        {
            "id": movie.get("id"),
            "title": movie.get("title"),
            "poster_path": movie.get("poster_path"),
            "vote_average": movie.get("vote_average"),
            "release_date": movie.get("release_date"),
        }
    )


def remove_from_watchlist(movie_id: int) -> None:
    _ensure_watchlist_state()
    st.session_state.watchlist = [m for m in st.session_state.watchlist if int(m["id"]) != int(movie_id)]


@st.cache_data(ttl=900, show_spinner=False)
def collect_candidate_movie_ids(max_titles: int = 180) -> list[int]:
    """Candidate pool from trending + several popular pages (deduped)."""
    ids: list[int] = []
    seen: set[int] = set()
    blocks: list[list[dict[str, Any]]] = [api.get_trending_movies("week")]
    for p in range(1, 6):
        blocks.append(api.get_popular_movies(p))
    for src in blocks:
        for item in src:
            rid = item.get("id")
            if isinstance(rid, int) and rid not in seen:
                seen.add(rid)
                ids.append(rid)
            if len(ids) >= max_titles:
                return ids
    return ids


@st.cache_data(ttl=900, show_spinner=False)
def load_candidate_corpus(max_titles: int = 180) -> list[dict[str, Any]]:
    """Fully hydrated TMDB records for offline-style ranking in-session."""
    out: list[dict[str, Any]] = []
    for mid in collect_candidate_movie_ids(max_titles):
        m = api.get_movie_details(mid)
        if m:
            out.append(m)
    return out


@st.cache_data(ttl=900, show_spinner=False)
def discover_hydrated(
    genre_ids: tuple[int, ...],
    *,
    year_min: int | None,
    year_max: int | None,
    min_rating: float | None,
    min_vote_count: int | None,
    runtime_min: int | None,
    runtime_max: int | None,
    language: str | None,
    pages: int = 5,
    limit: int = 220,
    genre_joiner: str = ",",
    sort_by: str = "popularity.desc",
) -> list[dict[str, Any]]:
    """
    Discover candidates from TMDB with strict constraints, then hydrate full details
    so we can apply local constraints that discover doesn't support (cast/director/cert).
    """
    # Note: we allow empty genre_ids when user only has a mood preference.
    discover_kwargs: dict[str, Any] = {
        "sort_by": sort_by,
        "with_genres": genre_joiner.join(str(g) for g in genre_ids) if genre_ids else None,
        "primary_release_date.gte": f"{year_min}-01-01" if year_min else None,
        "primary_release_date.lte": f"{year_max}-12-31" if year_max else None,
        "vote_average.gte": min_rating,
        "vote_count.gte": min_vote_count,
        "with_original_language": language,
        # runtime is enforced only when user changed it (see MovieFilters.enforce_runtime)
        "with_runtime.gte": runtime_min,
        "with_runtime.lte": runtime_max,
    }
    ids: list[int] = []
    seen: set[int] = set()
    for p in range(1, max(1, int(pages)) + 1):
        page = api.discover_movies_page(p, **discover_kwargs)
        for item in page:
            mid = item.get("id")
            if isinstance(mid, int) and mid not in seen:
                seen.add(mid)
                ids.append(mid)
        if len(ids) >= int(limit):
            break
    hydrated: list[dict[str, Any]] = []
    for mid in ids:
        m = api.get_movie_details(mid)
        if m:
            hydrated.append(m)
    return hydrated


@st.cache_data(ttl=900, show_spinner=False)
def hydrate_movie_details_batch(movie_ids: tuple[int, ...], limit: int = 220) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for mid in movie_ids[: max(1, int(limit))]:
        m = api.get_movie_details(int(mid))
        if m:
            out.append(m)
    return out


@st.cache_data(ttl=900, show_spinner=False)
def fetch_anchor_candidate_pool(anchor_id: int, pages: int = 3) -> list[dict[str, Any]]:
    """
    Similar page retrieval is anchor-first:
    1) /similar
    2) /recommendations
    3) optional fallback discover by anchor genres
    """
    ids: list[int] = []
    seen: set[int] = set()
    anchor = api.get_movie_details(anchor_id)
    # 0) same collection members first (strongest series continuity)
    if anchor:
        col = anchor.get("belongs_to_collection")
        if isinstance(col, dict) and col.get("id"):
            cdata = api.get_collection_details(int(col["id"]))
            if isinstance(cdata, dict):
                for row in cdata.get("parts", []) or []:
                    mid = row.get("id")
                    if isinstance(mid, int) and mid not in seen and mid != anchor_id:
                        seen.add(mid)
                        ids.append(mid)

    for p in range(1, max(1, int(pages)) + 1):
        for row in api.get_similar_movies(anchor_id, p):
            mid = row.get("id")
            if isinstance(mid, int) and mid not in seen and mid != anchor_id:
                seen.add(mid)
                ids.append(mid)
        for row in api.get_movie_recommendations(anchor_id, p):
            mid = row.get("id")
            if isinstance(mid, int) and mid not in seen and mid != anchor_id:
                seen.add(mid)
                ids.append(mid)

    if anchor and len(ids) < 80:
        top_genres = [g.get("id") for g in (anchor.get("genres") or []) if isinstance(g, dict) and g.get("id")]
        if top_genres:
            for p in range(1, 4):
                for row in api.discover_movies_page(
                    p,
                    with_genres=",".join(str(x) for x in top_genres[:3]),
                    sort_by="popularity.desc",
                ):
                    mid = row.get("id")
                    if isinstance(mid, int) and mid not in seen and mid != anchor_id:
                        seen.add(mid)
                        ids.append(mid)

    return hydrate_movie_details_batch(tuple(ids), limit=320)


@st.cache_data(ttl=900, show_spinner=False)
def build_anchor_relationship_context(anchor_id: int, pages: int = 3) -> dict[str, Any]:
    """Reusable relationship context for scoring/explanations."""
    similar_ids: set[int] = set()
    recommendation_ids: set[int] = set()
    collection_member_ids: set[int] = set()
    anchor = api.get_movie_details(anchor_id)
    if anchor:
        col = anchor.get("belongs_to_collection")
        if isinstance(col, dict) and col.get("id"):
            cdata = api.get_collection_details(int(col["id"]))
            if isinstance(cdata, dict):
                for row in cdata.get("parts", []) or []:
                    mid = row.get("id")
                    if isinstance(mid, int) and mid != anchor_id:
                        collection_member_ids.add(mid)
    for p in range(1, max(1, int(pages)) + 1):
        for row in api.get_similar_movies(anchor_id, p):
            mid = row.get("id")
            if isinstance(mid, int) and mid != anchor_id:
                similar_ids.add(mid)
        for row in api.get_movie_recommendations(anchor_id, p):
            mid = row.get("id")
            if isinstance(mid, int) and mid != anchor_id:
                recommendation_ids.add(mid)
    return {
        "anchor_id": int(anchor_id),
        "collection_member_ids": collection_member_ids,
        "similar_ids": similar_ids,
        "recommendation_ids": recommendation_ids,
    }


def _merge_movies_unique(*movie_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for lst in movie_lists:
        for m in lst or []:
            mid = m.get("id")
            if isinstance(mid, int) and mid not in out:
                out[mid] = m
    return list(out.values())


def infer_query_prototype_anchor(query_text: str) -> dict[str, Any] | None:
    """
    Best-effort inferred anchor from query text.
    Used to enrich Smart pools with TMDB relationship graph.
    """
    q = (query_text or "").strip()
    if not q:
        return None
    rows = api.search_movies(q, page=1)
    if not rows:
        return None
    # Pick the best high-confidence candidate from top search rows.
    q_terms = [t.lower() for t in q.split() if t.strip()]
    best_id: int | None = None
    best_score = -1.0
    for r in rows[:8]:
        title = str(r.get("title") or "").lower()
        if not title:
            continue
        t_terms = [t for t in title.replace(":", " ").replace("-", " ").split() if t]
        overlap = 0.0
        for qt in q_terms:
            if any(tt.startswith(qt[:4]) or qt.startswith(tt[:4]) for tt in t_terms if len(tt) >= 3 and len(qt) >= 3):
                overlap += 1.0
        overlap = overlap / max(1.0, float(len(q_terms)))
        pop = float(r.get("popularity") or 0.0)
        score = (0.7 * overlap) + (0.3 * min(1.0, pop / 100.0))
        if score > best_score and isinstance(r.get("id"), int):
            best_score = score
            best_id = int(r["id"])
    if best_id is None or best_score < 0.30:
        return None
    return api.get_movie_details(best_id)


def get_default_movie_filters_dict() -> dict[str, Any]:
    """Canonical filter snapshot stored in st.session_state['movie_filters']."""
    return {
        "era": "Any",
        "y_min": 1990,
        "y_max": 2025,
        "min_rating": 6.0,
        "min_votes": 200,
        "rt_min": 70,
        "rt_max": 200,
        "cert": "Any",
        "audience": "Any",
        "language": "",
        "actor_q": "",
        "director_q": "",
        "studio_q": "",
        "country_q": "",
        "keyword_q": "",
        "genre_mode": "OR",
        "quick_picks": [],
        "picked": [],
        "ex_anim": False,
        "ex_doc": False,
        "ex_horror": False,
        "ex_adult": True,
        "sort_by": "Best match",
    }


def _clear_prefixed_widget_keys(prefix: str) -> None:
    """Remove widget session keys so the next render re-inits from movie_filters dict."""
    pfx = f"{prefix}_"
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(pfx):
            del st.session_state[k]


def sync_shared_state_to_prefix(prefix: str) -> None:
    """
    Copy canonical movie_filters dict into prefixed widget keys only when those keys
    are missing — avoids overwriting in-flight widget edits on rerun.
    """
    d = st.session_state.setdefault("movie_filters", get_default_movie_filters_dict())
    if f"{prefix}_era" in st.session_state:
        return
    st.session_state[f"{prefix}_era"] = d["era"]
    st.session_state[f"{prefix}_y_min"] = int(d["y_min"])
    st.session_state[f"{prefix}_y_max"] = int(d["y_max"])
    st.session_state[f"{prefix}_min_rating"] = float(d["min_rating"])
    st.session_state[f"{prefix}_min_votes"] = int(d["min_votes"])
    st.session_state[f"{prefix}_rt"] = (int(d["rt_min"]), int(d["rt_max"]))
    st.session_state[f"{prefix}_cert"] = d["cert"]
    st.session_state[f"{prefix}_audience"] = d["audience"]
    st.session_state[f"{prefix}_language"] = d["language"]
    st.session_state[f"{prefix}_actor_q"] = d["actor_q"]
    st.session_state[f"{prefix}_director_q"] = d["director_q"]
    st.session_state[f"{prefix}_studio_q"] = d["studio_q"]
    st.session_state[f"{prefix}_country_q"] = d["country_q"]
    st.session_state[f"{prefix}_keyword_q"] = d["keyword_q"]
    st.session_state[f"{prefix}_genre_mode"] = d["genre_mode"]
    st.session_state[f"{prefix}_quick_picks"] = copy.deepcopy(d["quick_picks"])
    st.session_state[f"{prefix}_picked"] = copy.deepcopy(d["picked"])
    st.session_state[f"{prefix}_ex_anim"] = bool(d["ex_anim"])
    st.session_state[f"{prefix}_ex_doc"] = bool(d["ex_doc"])
    st.session_state[f"{prefix}_ex_horror"] = bool(d["ex_horror"])
    st.session_state[f"{prefix}_ex_adult"] = bool(d["ex_adult"])
    st.session_state[f"{prefix}_sort_by"] = d["sort_by"]


def save_prefix_to_shared(prefix: str) -> None:
    """Persist prefixed widget keys into canonical movie_filters (session persistence)."""
    rt = st.session_state[f"{prefix}_rt"]
    if isinstance(rt, (list, tuple)) and len(rt) == 2:
        rt_min = int(rt[0])
        rt_max = int(rt[1])
    elif isinstance(rt, (int, float)):
        # Corrupted widget state can surface as a scalar; recover to safe defaults
        # instead of collapsing the range to a mirrored single value.
        rt_min, rt_max = 70, 200
    else:
        rt_min, rt_max = 70, 200
    d = st.session_state.setdefault("movie_filters", get_default_movie_filters_dict())
    d["era"] = st.session_state[f"{prefix}_era"]
    d["y_min"] = int(st.session_state[f"{prefix}_y_min"])
    d["y_max"] = int(st.session_state[f"{prefix}_y_max"])
    d["min_rating"] = float(st.session_state[f"{prefix}_min_rating"])
    d["min_votes"] = int(st.session_state[f"{prefix}_min_votes"])
    d["rt_min"] = rt_min
    d["rt_max"] = rt_max
    d["cert"] = st.session_state[f"{prefix}_cert"]
    d["audience"] = st.session_state[f"{prefix}_audience"]
    d["language"] = str(st.session_state[f"{prefix}_language"])
    d["actor_q"] = str(st.session_state[f"{prefix}_actor_q"])
    d["director_q"] = str(st.session_state[f"{prefix}_director_q"])
    d["studio_q"] = str(st.session_state[f"{prefix}_studio_q"])
    d["country_q"] = str(st.session_state[f"{prefix}_country_q"])
    d["keyword_q"] = str(st.session_state[f"{prefix}_keyword_q"])
    d["genre_mode"] = st.session_state[f"{prefix}_genre_mode"]
    d["quick_picks"] = copy.deepcopy(st.session_state[f"{prefix}_quick_picks"])
    d["picked"] = copy.deepcopy(st.session_state[f"{prefix}_picked"])
    d["ex_anim"] = bool(st.session_state[f"{prefix}_ex_anim"])
    d["ex_doc"] = bool(st.session_state[f"{prefix}_ex_doc"])
    d["ex_horror"] = bool(st.session_state[f"{prefix}_ex_horror"])
    d["ex_adult"] = bool(st.session_state[f"{prefix}_ex_adult"])
    d["sort_by"] = st.session_state[f"{prefix}_sort_by"]
    if prefix == "smart":
        _clear_prefixed_widget_keys("sim")
    elif prefix == "sim":
        _clear_prefixed_widget_keys("smart")


def _dict_to_movie_filters(d: dict[str, Any], genre_options: list[dict[str, Any]]) -> filter_mod.MovieFilters:
    """Build MovieFilters from a canonical filter dict (same semantics as the former sidebar)."""
    era = d["era"]
    y_min = int(d["y_min"])
    y_max = int(d["y_max"])
    if ERA_MAP.get(era):
        y_min, y_max = ERA_MAP[era]  # type: ignore[assignment]
    min_rating = float(d["min_rating"])
    min_votes = int(d["min_votes"])
    rt1 = int(d["rt_min"])
    rt2 = int(d["rt_max"])
    cert = d["cert"]
    audience = d["audience"]
    language = str(d["language"] or "")
    actor_q = str(d["actor_q"] or "")
    director_q = str(d["director_q"] or "")
    studio_q = str(d["studio_q"] or "")
    country_q = str(d["country_q"] or "")
    keyword_q = str(d["keyword_q"] or "")
    genre_mode = d["genre_mode"]
    quick_picks = list(d.get("quick_picks") or [])
    picked = list(d.get("picked") or [])
    ex_anim = bool(d["ex_anim"])
    ex_doc = bool(d["ex_doc"])
    ex_horror = bool(d["ex_horror"])
    ex_adult = bool(d["ex_adult"])
    sort_by = d["sort_by"]

    default_year = (1990, 2025)
    year_active = (int(y_min), int(y_max)) != default_year
    vote_count_active = min_votes >= 500
    default_rt = (70, 200)
    enforce_runtime = (rt1, rt2) != default_rt

    name_to_id = {g["name"]: g["id"] for g in genre_options if g.get("name") and g.get("id")}
    merged_picks = sorted(list({*picked, *quick_picks}))

    return filter_mod.MovieFilters(
        year_min=int(y_min),
        year_max=int(y_max),
        min_rating=float(min_rating),
        min_vote_count=int(min_votes),
        runtime_min=int(rt1),
        runtime_max=int(rt2),
        enforce_runtime=enforce_runtime,
        year_active=year_active,
        vote_count_active=vote_count_active,
        certification=None if cert == "Any" else cert,
        audience_level=None if audience == "Any" else audience,
        language=language.strip() or None,
        actor_names=[s.strip() for s in actor_q.split(",") if s.strip()],
        director_names=[s.strip() for s in director_q.split(",") if s.strip()],
        studio_contains=studio_q.strip() or None,
        country_contains=country_q.strip() or None,
        keyword_contains=keyword_q.strip() or None,
        exclude_animation=ex_anim,
        exclude_documentary=ex_doc,
        exclude_horror=ex_horror,
        exclude_adult=ex_adult,
        sort_by=sort_by,
        genre_ids=[name_to_id[n] for n in merged_picks if n in name_to_id],
        genre_mode=cast_genre_mode(genre_mode),
    )


def movie_filters_from_shared_dict(genre_options: list[dict[str, Any]]) -> filter_mod.MovieFilters:
    """Insights and any page that reads the canonical dict after Smart/Similar have saved."""
    d = st.session_state.setdefault("movie_filters", get_default_movie_filters_dict())
    return _dict_to_movie_filters(d, genre_options)


def movie_filters_from_prefix(prefix: str, genre_options: list[dict[str, Any]]) -> filter_mod.MovieFilters:
    """Build MovieFilters from current prefixed widget state (after sync + render)."""
    rt = st.session_state[f"{prefix}_rt"]
    if isinstance(rt, (list, tuple)) and len(rt) == 2:
        rt_min = int(rt[0])
        rt_max = int(rt[1])
    elif isinstance(rt, (int, float)):
        rt_min, rt_max = 70, 200
    else:
        rt_min, rt_max = 70, 200
    d = {
        "era": st.session_state[f"{prefix}_era"],
        "y_min": st.session_state[f"{prefix}_y_min"],
        "y_max": st.session_state[f"{prefix}_y_max"],
        "min_rating": st.session_state[f"{prefix}_min_rating"],
        "min_votes": st.session_state[f"{prefix}_min_votes"],
        "rt_min": rt_min,
        "rt_max": rt_max,
        "cert": st.session_state[f"{prefix}_cert"],
        "audience": st.session_state[f"{prefix}_audience"],
        "language": st.session_state[f"{prefix}_language"],
        "actor_q": st.session_state[f"{prefix}_actor_q"],
        "director_q": st.session_state[f"{prefix}_director_q"],
        "studio_q": st.session_state[f"{prefix}_studio_q"],
        "country_q": st.session_state[f"{prefix}_country_q"],
        "keyword_q": st.session_state[f"{prefix}_keyword_q"],
        "genre_mode": st.session_state[f"{prefix}_genre_mode"],
        "quick_picks": st.session_state[f"{prefix}_quick_picks"],
        "picked": st.session_state[f"{prefix}_picked"],
        "ex_anim": st.session_state[f"{prefix}_ex_anim"],
        "ex_doc": st.session_state[f"{prefix}_ex_doc"],
        "ex_horror": st.session_state[f"{prefix}_ex_horror"],
        "ex_adult": st.session_state[f"{prefix}_ex_adult"],
        "sort_by": st.session_state[f"{prefix}_sort_by"],
    }
    return _dict_to_movie_filters(d, genre_options)


def render_filter_widgets(genre_options: list[dict[str, Any]], key_prefix: str) -> None:
    """All filter controls (used inside popover or expander)."""
    p = key_prefix
    name_to_id = {g["name"]: g["id"] for g in genre_options if g.get("name") and g.get("id")}
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Release era", list(ERA_MAP.keys()), key=f"{p}_era")
    with c2:
        st.selectbox(
            "Certification (US)",
            ["Any", "G", "PG", "PG-13", "R", "NC-17"],
            key=f"{p}_cert",
        )
    y_col1, y_col2 = st.columns(2)
    with y_col1:
        st.number_input("Year min", 1920, 2035, key=f"{p}_y_min")
    with y_col2:
        st.number_input("Year max", 1920, 2035, key=f"{p}_y_max")
    st.slider("Minimum rating", 0.0, 10.0, key=f"{p}_min_rating", step=0.5)
    st.number_input("Minimum vote count", 0, 50_000, key=f"{p}_min_votes", step=50)
    st.slider("Runtime (min)", 40, 220, key=f"{p}_rt")
    st.selectbox("Audience level", ["Any", "Kids / Family", "Teen", "Mature"], key=f"{p}_audience")
    st.text_input("Language (ISO 639-1)", key=f"{p}_language", placeholder="e.g. en")
    a1, a2 = st.columns(2)
    with a1:
        st.text_input("Actor contains", key=f"{p}_actor_q", placeholder="comma-separated")
    with a2:
        st.text_input("Director contains", key=f"{p}_director_q", placeholder="comma-separated")
    s1, s2 = st.columns(2)
    with s1:
        st.text_input("Studio contains", key=f"{p}_studio_q", placeholder="Pixar, Disney, A24...")
    with s2:
        st.text_input("Production country contains", key=f"{p}_country_q", placeholder="US, Japan, UK...")
    st.text_input("Keyword contains", key=f"{p}_keyword_q", placeholder="superhero, heist, time travel...")
    st.radio("Genre match", ["OR", "AND"], horizontal=True, key=f"{p}_genre_mode")
    g1, g2 = st.columns(2)
    with g1:
        st.multiselect("Quick genres", [g for g in QUICK_GENRES if g in name_to_id], key=f"{p}_quick_picks")
    with g2:
        st.multiselect("Genres", sorted(name_to_id.keys()), key=f"{p}_picked")
    st.markdown("##### Exclude")
    ex1, ex2 = st.columns(2)
    with ex1:
        st.checkbox("Exclude animation", key=f"{p}_ex_anim")
        st.checkbox("Exclude documentary", key=f"{p}_ex_doc")
    with ex2:
        st.checkbox("Exclude horror", key=f"{p}_ex_horror")
        st.checkbox("Exclude adult titles", key=f"{p}_ex_adult")
    st.selectbox(
        "Sort results",
        ["Best match", "Highest rated", "Most popular", "Newest", "Shortest runtime", "Longest runtime"],
        key=f"{p}_sort_by",
    )


def _render_filters_dialog(genre_options: list[dict[str, Any]], key_prefix: str, title: str, visibility_key: str) -> None:
    """Render filters inside a modal dialog when runtime supports st.dialog."""
    if not hasattr(st, "dialog"):
        st.error("This Streamlit runtime does not support modal dialogs. Please upgrade Streamlit.")
        st.session_state[visibility_key] = False
        return

    @st.dialog(title, width="large")
    def _filters_dialog() -> None:
        st.markdown('<div class="pf-filter-modal-shell">', unsafe_allow_html=True)
        render_filter_widgets(genre_options, key_prefix)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Apply filters", key=f"{key_prefix}_filters_apply", type="primary"):
                save_prefix_to_shared(key_prefix)
                st.session_state[visibility_key] = False
                st.rerun()
        with c2:
            if st.button("Close", key=f"{key_prefix}_filters_close"):
                st.session_state[visibility_key] = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    _filters_dialog()


def _open_modal_flag(flag_key: str) -> None:
    st.session_state[flag_key] = True


def _close_modal_flag(flag_key: str) -> None:
    st.session_state[flag_key] = False


def _on_similar_anchor_change() -> None:
    # Resolve selected label -> TMDB movie_id from persisted search mapping.
    label = st.session_state.get("similar_anchor_select")
    mapping = st.session_state.get("similar_search_label_to_id", {})
    selected_id: int | None = None
    if isinstance(label, str) and isinstance(mapping, dict):
        raw_id = mapping.get(label)
        if isinstance(raw_id, int):
            selected_id = int(raw_id)
    st.session_state["similar_selected_movie_id"] = selected_id
    # Selecting an anchor is an explicit "next step" signal; run Similar automatically.
    st.session_state["show_similar_filters"] = False
    st.session_state["sim_auto_compute"] = bool(selected_id)


def render_movie_filters_panel(
    genre_options: list[dict[str, Any]],
    key_prefix: str,
    *,
    visibility_key: str,
    toggle_button_key: str,
    title: str,
) -> None:
    """
    Renders a page-level toggle button and modal filter dialog on demand.
    Each page must call save_prefix_to_shared(key_prefix) after widgets are used so
    movie_filters stays canonical and the other tab's keys can be invalidated.
    """
    sync_shared_state_to_prefix(key_prefix)
    if visibility_key not in st.session_state:
        st.session_state[visibility_key] = False
    st.button(
        "Filters",
        key=toggle_button_key,
        on_click=_open_modal_flag,
        args=(visibility_key,),
    )
    if st.session_state[visibility_key]:
        _render_filters_dialog(genre_options, key_prefix, title, visibility_key)


def cast_genre_mode(label: str) -> filter_mod.GenreMode:
    return "AND" if label == "AND" else "OR"


def build_family_safe_adjacent_genres(selected_genre_ids: list[int]) -> list[int]:
    """
    Map mature-adjacent genre requests to family-safe adjacent options.
    Used only for Kids / Family conflict fallback mode.
    """
    out: set[int] = set()
    # Always keep core family-safe anchors.
    out.update({10751, 16, 12, 14, 35, 9648})  # Family/Animation/Adventure/Fantasy/Comedy/Mystery
    selected = set(int(x) for x in (selected_genre_ids or []))
    if 80 in selected:  # Crime
        out.update({9648, 12, 35, 10751, 16})
    if 53 in selected:  # Thriller
        out.update({9648, 12, 14, 10751})
    if 10752 in selected:  # War
        out.update({12, 36, 18, 10751})  # Adventure/History/Drama/Family
    if 27 in selected:  # Horror
        out.update({14, 9648, 12, 10751})  # Fantasy/Mystery/Adventure/Family
    return sorted(out)


def sort_ranked_results(
    ranked: list[tuple[dict[str, Any], float, dict[str, float]]],
    sort_by: str,
) -> list[tuple[dict[str, Any], float, dict[str, float]]]:
    if sort_by == "Best match":
        return ranked
    if sort_by == "Highest rated":
        return sorted(ranked, key=lambda x: float(x[0].get("vote_average") or 0.0), reverse=True)
    if sort_by == "Most popular":
        return sorted(ranked, key=lambda x: float(x[0].get("popularity") or 0.0), reverse=True)
    if sort_by == "Newest":
        return sorted(ranked, key=lambda x: str(x[0].get("release_date") or ""), reverse=True)
    if sort_by == "Shortest runtime":
        return sorted(ranked, key=lambda x: int(x[0].get("runtime") or 9999))
    if sort_by == "Longest runtime":
        return sorted(ranked, key=lambda x: int(x[0].get("runtime") or 0), reverse=True)
    return ranked


def apply_quality_gate(movies: list[dict[str, Any]], *, min_votes: int, min_rating: float = 5.0) -> list[dict[str, Any]]:
    """
    Stabilize Smart results by preferring titles with enough social proof.
    Falls back to original list if gate is too restrictive.
    """
    gated = [
        m
        for m in movies
        if int(m.get("vote_count") or 0) >= int(min_votes) and float(m.get("vote_average") or 0.0) >= float(min_rating)
    ]
    return gated if len(gated) >= 5 else movies


def try_api_or_stop() -> bool:
    token = os.environ.get("TMDB_API_KEY", "").strip()
    if not token:
        st.error(
            "Missing **TMDB_API_KEY**. Add your TMDB v3 API key or Read Access Token to the "
            "environment or a `.env` file (see `.env.example`)."
        )
        st.stop()
    return True


def apply_custom_theme_css() -> None:
    st.markdown(
        """
<style>

/* ===== Background ===== */
html, body, .stApp {
    background: linear-gradient(
        135deg,
        #ffffff 0%,
        #ffe4ec 30%,
        #ffd1dc 55%,
        #ffc2a1 80%,
        #ffffff 100%
    );
}

/* ===== FILTER MODAL THEME ===== */
:root {
    --pf-orange: #f6b7a3;
    --pf-orange-dark: #ee9d86;
    --pf-input: #fffaf8;
    --pf-text: #111111;
}

/* Modal overlay */
div[data-testid="stDialog"] {
    background: rgba(246, 183, 163, 0.35) !important;
}

/* Actual filter modal wrapper used by live render path */
.pf-filter-modal-shell {
    background: #f6b7a3 !important;
    color: black !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 18px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12) !important;
    padding: 12px !important;
}

.pf-filter-modal-shell * {
    color: black !important;
}

.pf-filter-modal-shell input,
.pf-filter-modal-shell textarea,
.pf-filter-modal-shell select,
.pf-filter-modal-shell div[data-baseweb="select"] > div,
.pf-filter-modal-shell div[data-testid="stNumberInput"] input {
    background: #fffaf8 !important;
    color: #111111 !important;
}

/* Force the ENTIRE popup content area to orange */
div[data-testid="stDialog"] [data-testid="stAppViewContainer"],
div[data-testid="stDialog"] .main,
div[data-testid="stDialog"] section.main,
div[data-testid="stDialog"] div.block-container {
    background: #f6b7a3 !important;
    color: black !important;
}

/* Remove dark backgrounds from all nested containers */
div[data-testid="stDialog"] div {
    background-color: transparent !important;
}

/* Keep inputs readable */
div[data-testid="stDialog"] input,
div[data-testid="stDialog"] textarea,
div[data-testid="stDialog"] div[data-baseweb="select"] > div {
    background: white !important;
    color: black !important;
}

/* Labels + text */
div[data-testid="stDialog"] label,
div[data-testid="stDialog"] p,
div[data-testid="stDialog"] span,
div[data-testid="stDialog"] h1,
div[data-testid="stDialog"] h2,
div[data-testid="stDialog"] h3 {
    color: black !important;
}

/* Fix number input +/- buttons */
div[data-testid="stDialog"] button[kind="secondary"],
div[data-testid="stDialog"] button[kind="secondary"]:hover {
    background: white !important;
    color: black !important;
    border: 1px solid rgba(0,0,0,0.15) !important;
}

/* Specifically target number input button group */
div[data-testid="stDialog"] div[data-testid="stNumberInput"] button {
    background: white !important;
    color: black !important;
    border-left: 1px solid rgba(0,0,0,0.15) !important;
}

/* Make sure the whole input group blends */
div[data-testid="stDialog"] div[data-testid="stNumberInput"] {
    background: transparent !important;
}

/* Actual modal panel */
div[data-testid="stDialog"] > div {
    background: var(--pf-orange) !important;
    color: var(--pf-text) !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 18px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12) !important;
}

/* Everything inside the modal defaults to dark text */
div[data-testid="stDialog"] * {
    color: var(--pf-text) !important;
}

/* The inner Streamlit block containers inside the dialog */
div[data-testid="stDialog"] [data-testid="stVerticalBlock"],
div[data-testid="stDialog"] [data-testid="stHorizontalBlock"],
div[data-testid="stDialog"] [data-testid="element-container"],
div[data-testid="stDialog"] [data-testid="stForm"] {
    background: transparent !important;
    color: var(--pf-text) !important;
}

/* Labels */
div[data-testid="stDialog"] label,
div[data-testid="stDialog"] p,
div[data-testid="stDialog"] span,
div[data-testid="stDialog"] h1,
div[data-testid="stDialog"] h2,
div[data-testid="stDialog"] h3,
div[data-testid="stDialog"] h4 {
    color: var(--pf-text) !important;
}

/* Text inputs / number inputs / text areas */
div[data-testid="stDialog"] input,
div[data-testid="stDialog"] textarea {
    background: var(--pf-input) !important;
    color: var(--pf-text) !important;
    border: 1px solid rgba(0,0,0,0.15) !important;
    border-radius: 10px !important;
}

/* Selectbox / multiselect control */
div[data-testid="stDialog"] div[data-baseweb="select"] > div {
    background: var(--pf-input) !important;
    color: var(--pf-text) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0,0,0,0.15) !important;
}

/* Dropdown menu */
div[role="listbox"],
ul[role="listbox"] {
    background: var(--pf-input) !important;
    color: var(--pf-text) !important;
}

/* Buttons inside modal */
div[data-testid="stDialog"] button {
    border-radius: 10px !important;
}

/* Keep your orange action buttons */
div[data-testid="stDialog"] .stButton > button,
div[data-testid="stDialog"] button[kind="primary"],
div[data-testid="stDialog"] button[kind="secondary"] {
    background: var(--pf-orange-dark) !important;
    color: var(--pf-text) !important;
    border: none !important;
}

/* Checkbox / radio text */
div[data-testid="stDialog"] [data-testid="stCheckbox"],
div[data-testid="stDialog"] [role="radiogroup"] {
    color: var(--pf-text) !important;
}

/* Slider labels */
div[data-testid="stDialog"] .stSlider {
    color: var(--pf-text) !important;
}

/* ===== GLOBAL TEXT ===== */
html, body, [class*="css"] {
    color: black !important;
}

/* ===== Headings ===== */
h1, h2, h3, h4 {
    color: #000000 !important;
    font-weight: 700;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.95);
}

/* ===== Fix dropdown text ===== */
.stSelectbox div, .stMultiSelect div {
    color: #000000 !important;
}

/* ===== Buttons ===== */
.stButton > button {
    background: linear-gradient(135deg, #ff4d6d, #ff7a5c);
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

/* ===== Tabs (top nav) ===== */
button[data-baseweb="tab"] {
    color: #333 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #ff4d6d !important;
    font-weight: 600;
}

/* ===== Alerts ===== */
.stAlert {
    color: #000 !important;
}

/* ===== Fix faded placeholder text ===== */
::placeholder {
    color: #666 !important;
}

/* ===== Remove washed-out look ===== */
[data-testid="stMarkdownContainer"] p {
    color: #111111 !important;
}

/* === YOUR ORANGE COLOR === */
:root {
    --main-orange: #ffb199;
}

/* === FILTER BUTTON === */
button[kind="secondary"], button[kind="primary"] {
    background: var(--main-orange) !important;
    color: black !important;
    border: none !important;
    font-weight: 600 !important;
}

/* === THE ACTUAL DARK PANEL FIX === */
/* This is the key fix */
div[data-testid="stPopover"] div[data-testid="stVerticalBlock"] {
    background: var(--main-orange) !important;
}

/* === ALSO FORCE ALL INNER BLOCKS === */
div[data-testid="stPopover"] div[data-testid="stVerticalBlock"] > div {
    background: var(--main-orange) !important;
}

/* === TEXT === */
div[data-testid="stPopover"] * {
    color: black !important;
}

/* === INPUTS (KEEP WHITE) === */
div[data-testid="stPopover"] input,
div[data-testid="stPopover"] textarea,
div[data-testid="stPopover"] select {
    background: white !important;
    color: black !important;
}

/* === DROPDOWNS === */
div[data-testid="stPopover"] div[data-baseweb="select"] {
    background: white !important;
    color: black !important;
}

/* === NUMBER INPUT === */
div[data-testid="stPopover"] div[data-testid="stNumberInput"] input {
    background: white !important;
    color: black !important;
}

/* === SLIDERS === */
div[data-testid="stPopover"] .stSlider {
    color: black !important;
}

/* === REMOVE DARK CARD EFFECT === */
div[data-testid="stPopover"] {
    box-shadow: none !important;
    border: none !important;
}

/* ===== EXPANDER FALLBACK (when st.popover unavailable) ===== */
div[data-testid="stExpander"] {
    background: var(--main-orange) !important;
    box-shadow: none !important;
    border: 1px solid rgba(0,0,0,0.1);
}
div[data-testid="stExpander"] > div {
    background: var(--main-orange) !important;
    color: black !important;
    border-radius: 14px !important;
    padding: 16px !important;
}
div[data-testid="stExpander"] * {
    background: var(--main-orange) !important;
    color: black !important;
}
div[data-testid="stExpander"] input,
div[data-testid="stExpander"] textarea,
div[data-testid="stExpander"] select {
    background: white !important;
    color: black !important;
}
div[data-testid="stExpander"] div[data-baseweb="select"] {
    background: white !important;
    color: black !important;
}
div[data-testid="stExpander"] div[data-testid="stNumberInput"] input {
    background: white !important;
    color: black !important;
}
div[data-testid="stExpander"] .stSlider label {
    color: black !important;
}
div[data-testid="stExpander"] button {
    background: var(--main-orange) !important;
    color: black !important;
    font-weight: 600 !important;
}

/* ===== INPUT FIELDS (rest of app) ===== */
input, textarea, select {
    background-color: #ffffff !important;
    color: black !important;
    border-radius: 8px !important;
}

/* ===== DROPDOWNS (rest of app) ===== */
div[data-baseweb="select"] {
    background-color: white !important;
    color: black !important;
}

/* ===== Similar anchor selectbox readability ===== */
.st-key-similar_anchor_select div[data-baseweb="select"] > div,
.st-key-similar_anchor_select [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #111111 !important;
    border: 1px solid rgba(0,0,0,0.20) !important;
    border-radius: 10px !important;
}
.st-key-similar_anchor_select div[data-baseweb="select"] span,
.st-key-similar_anchor_select div[data-baseweb="select"] input,
.st-key-similar_anchor_select [data-baseweb="select"] * {
    color: #111111 !important;
    fill: #111111 !important;
}
.st-key-similar_anchor_select [data-baseweb="select"] svg {
    color: #111111 !important;
    fill: #111111 !important;
}
ul[role="listbox"],
div[role="listbox"] {
    background: #ffffff !important;
    color: #111111 !important;
}
ul[role="listbox"] li,
div[role="listbox"] [role="option"] {
    background: #ffffff !important;
    color: #111111 !important;
}
ul[role="listbox"] li:hover,
div[role="listbox"] [role="option"]:hover {
    background: #ffe4ec !important;
    color: #111111 !important;
}

/* ===== NUMBER INPUT (rest of app) ===== */
div[data-testid="stNumberInput"] input {
    background-color: white !important;
    color: black !important;
}

/* ===== SLIDER LABELS (rest of app) ===== */
.stSlider label {
    color: black !important;
}

</style>
""",
        unsafe_allow_html=True,
    )


def page_home() -> None:
    st.markdown(
        """
<h1 style='color:#000000; font-weight:800;'>PickAFlick</h1>
""",
        unsafe_allow_html=True,
    )
    film_strip = Path("your_film_strip_image.png")
    if film_strip.exists():
        st.image(str(film_strip), use_column_width=True)
    st.markdown(
        "Discover what to watch next with **content-based TF-IDF**, **overview sentiment**, "
        "and **popularity-aware** blending — all powered by TMDB."
    )
    st.divider()
    st.subheader("Trending this week")
    try:
        rows = api.get_trending_movies("week")[:12]
    except RuntimeError as e:
        ui_components.error_bubble(str(e))
        return
    if not rows:
        ui_components.warning_bubble("No trending titles returned from TMDB.")
        return
    ui_components.movie_grid(rows, cols=4, key_prefix="home")


def page_smart(genre_options: list[dict[str, Any]]) -> None:
    st.header("Smart recommendations")
    _shown_info: set[str] = set()

    def _info_once(msg: str) -> None:
        if msg not in _shown_info:
            st.info(msg)
            _shown_info.add(msg)

    def _debug(msg: str) -> None:
        if SHOW_DEBUG:
            st.caption(msg)

    filt_col, _ = st.columns([1, 10])
    with filt_col:
        render_movie_filters_panel(
            genre_options,
            "smart",
            visibility_key="show_smart_filters",
            toggle_button_key="smart_filters_btn",
            title="Smart Recommendation Filters",
        )
    base_filters = movie_filters_from_prefix("smart", genre_options)
    st.markdown(
        "Describe the kind of movie you want, then optionally pick an anchor to bias results toward a specific title."
    )
    nl = st.text_area(
        "Describe what you want",
        height=88,
        placeholder='e.g. "funny action movie under 2 hours from the 2000s"',
    )
    parsed = nlp_utils.parse_natural_language_query(nl or "")
    active_filters = filter_mod.merge_parsed_nl_into_filters(
        filter_mod.MovieFilters(
            year_min=base_filters.year_min,
            year_max=base_filters.year_max,
            min_rating=base_filters.min_rating,
            min_vote_count=base_filters.min_vote_count,
            runtime_min=base_filters.runtime_min,
            runtime_max=base_filters.runtime_max,
            certification=base_filters.certification,
            language=base_filters.language,
            actor_names=list(base_filters.actor_names),
            director_names=list(base_filters.director_names),
            genre_ids=list(base_filters.genre_ids),
            genre_mode=base_filters.genre_mode,
        ),
        parsed,
    )
    # Carry exclusion and hard-safety flags into merged filters.
    active_filters.exclude_animation = base_filters.exclude_animation
    active_filters.exclude_documentary = base_filters.exclude_documentary
    active_filters.exclude_horror = base_filters.exclude_horror
    active_filters.exclude_adult = base_filters.exclude_adult
    active_filters.audience_level = base_filters.audience_level
    active_filters.allow_uncertified_in_fallback = base_filters.allow_uncertified_in_fallback
    active_filters.studio_contains = base_filters.studio_contains
    active_filters.country_contains = base_filters.country_contains
    active_filters.keyword_contains = base_filters.keyword_contains
    active_filters.language = base_filters.language
    active_filters.certification = base_filters.certification
    active_filters.sort_by = base_filters.sort_by
    active_filters.enforce_runtime = base_filters.enforce_runtime
    active_filters.year_active = base_filters.year_active
    active_filters.vote_count_active = base_filters.vote_count_active

    # Apply query-derived exclusions (e.g. "no superheroes", "without animation")
    active_filters.exclude_superhero = getattr(parsed, "exclude_superhero", False)
    active_filters.excluded_genre_ids = list(getattr(parsed, "excluded_genre_ids", []))

    target_n = 10
    if parsed.genre_names:
        st.caption("Parsed genres: " + ", ".join(parsed.genre_names))
    if parsed.mood:
        st.caption(f"Parsed mood hint: **{parsed.mood}**")

    st.subheader("Optional: Choose a movie to find similar titles")
    mode = st.radio(
        "Optional anchor via",
        ["None", "Search", "Pick from trending"],
        horizontal=True,
        key="smart_anchor_mode",
    )
    anchor: dict[str, Any] | None = None
    if mode == "Search":
        q = st.text_input("Search title", placeholder="Inception", key="smart_anchor_search_input")
        if st.button("Search", type="primary", key="smart_anchor_search_btn"):
            st.session_state["smart_search_q"] = q
        qkey = st.session_state.get("smart_search_q", "")
        if qkey:
            res = api.search_movies(qkey, page=1)
            if not res:
                st.warning("No search results.")
            else:
                labels = {f"{r['title']} ({r.get('release_date', '')[:4]})": r["id"] for r in res[:25]}
                choice = st.selectbox("Pick a title", list(labels.keys()), key="smart_anchor_select")
                aid = labels[choice]
                anchor = api.get_movie_details(int(aid))
    elif mode == "Pick from trending":
        trend = api.get_trending_movies("week")
        labels = {f"{r['title']} ({r.get('release_date', '')[:4]})": r["id"] for r in trend[:30]}
        if labels:
            choice = st.selectbox("Trending pick", list(labels.keys()), key="smart_trending_anchor_select")
            anchor = api.get_movie_details(int(labels[choice]))

    has_query_input = bool((nl or "").strip())
    has_genres = bool(active_filters.genre_ids)
    has_anchor_input = bool(anchor)
    inferred_anchor: dict[str, Any] | None = None
    if not anchor and has_query_input:
        inferred_anchor = infer_query_prototype_anchor((nl or "").strip())
    relationship_anchor = anchor or inferred_anchor
    relationship_context = (
        build_anchor_relationship_context(int(relationship_anchor.get("id")), pages=3)
        if relationship_anchor and relationship_anchor.get("id")
        else None
    )
    smart_sig = (
        tuple(sorted(active_filters.genre_ids)),
        active_filters.genre_mode,
        active_filters.year_min,
        active_filters.year_max,
        active_filters.year_active,
        active_filters.min_rating,
        active_filters.min_vote_count,
        active_filters.vote_count_active,
        active_filters.runtime_min,
        active_filters.runtime_max,
        active_filters.enforce_runtime,
        active_filters.certification,
        active_filters.audience_level,
        active_filters.language,
        active_filters.studio_contains,
        active_filters.country_contains,
        active_filters.keyword_contains,
        active_filters.exclude_animation,
        active_filters.exclude_documentary,
        active_filters.exclude_horror,
        active_filters.exclude_adult,
        active_filters.exclude_superhero,
        tuple(sorted(active_filters.excluded_genre_ids)),
        bool((nl or "").strip()),
        (anchor.get("id") if anchor else None),
        (relationship_anchor.get("id") if relationship_anchor else None),
    )
    if st.session_state.get("smart_filter_sig") != smart_sig:
        st.session_state.pop("smart_ranked", None)
        st.session_state["smart_filter_sig"] = smart_sig
    if not has_query_input and not has_genres and not has_anchor_input:
        st.info("Provide at least one signal: a genre, a description, or an optional anchor movie.")

    # Negative-only query guard: avoid broad retrieval when user only excludes (e.g. "no superheroes").
    if nlp_utils.is_negative_only_request(
        parsed,
        has_any_genres_selected=bool(active_filters.genre_ids),
        has_anchor=bool(anchor),
    ):
        st.warning(
            "I can avoid superhero-related titles. Add a genre, mood, keyword, or anchor movie so I can narrow recommendations."
        )
        st.session_state["smart_ranked"] = (anchor, [])
        save_prefix_to_shared("smart")
        return

    if st.button("Generate hybrid recommendations", type="primary"):
        with st.spinner("Loading candidate corpus + ranking…"):
            # HARD ORDER:
            # 1) parse -> merged already
            # 2) validate at least one genre exists (manual or parsed)
            if not has_query_input and not has_genres and not has_anchor_input:
                st.warning("Please select at least one genre, add a description, or choose an anchor movie.")
                st.session_state["smart_ranked"] = (anchor, [])
            else:
                conflicts = filter_mod.detect_filter_conflicts(active_filters)
                safe_compromise = False
                family_conflict_fallback = False
                if conflicts:
                    st.warning(
                        "Your filters are conflicting. Kids / Family titles rarely overlap with Crime or mature ratings."
                    )
                    if (active_filters.audience_level or "").lower() == "kids / family":
                        # Hard stop only for truly incompatible combinations.
                        if (active_filters.certification or "").upper() in {"NC-17"}:
                            st.warning("Please relax certification for Kids / Family content.")
                            st.session_state["smart_ranked"] = (anchor, [])
                            save_prefix_to_shared("smart")
                            return
                        family_conflict_fallback = True

                # 3) retrieval-first strategy. For explicit multi-genre requests, build:
                #    A) exact AND-genre pool first
                #    B) broader OR-genre fallback pool only if exact is sparse
                mood_tgt = nlp_utils.parsed_mood_target_sentiment(parsed.mood)
                query_terms = nlp_utils.extract_query_terms(nl or "")
                requested_genres = tuple(sorted(active_filters.genre_ids))
                require_superhero = bool(getattr(parsed, "require_superhero_theme", False))
                require_animal_theme = bool(getattr(parsed, "require_animal_theme", False))
                require_music_theme = bool(getattr(parsed, "require_music_theme", False))
                require_romance = (10749 in set(requested_genres)) or ("romantic" in (nl or "").lower())
                required_franchise = getattr(parsed, "required_franchise", None)
                # For superhero-themed queries, broaden retrieval beyond Romance-only pools.
                # Theme gating below will enforce superhero relevance.
                superhero_retrieval_genres = (28, 12, 14, 878)  # Action/Adventure/Fantasy/Sci-Fi
                discover_genres = (
                    tuple(sorted(set(requested_genres) | set(superhero_retrieval_genres)))
                    if require_superhero
                    else requested_genres
                )
                animal_retrieval_genres = (10751, 16, 12, 14, 35)  # Family/Animation/Adventure/Fantasy/Comedy
                if require_animal_theme:
                    discover_genres = tuple(sorted(set(discover_genres) | set(animal_retrieval_genres)))
                music_retrieval_genres = (10402, 16, 10751, 14, 35)  # Music + family musical adjacencies
                if require_music_theme:
                    discover_genres = tuple(sorted(set(discover_genres) | set(music_retrieval_genres)))
                # Franchise queries need a larger pool; we hard-filter to DC/Marvel after hydration.
                if required_franchise:
                    discover_genres = tuple(sorted(set(discover_genres) | set(superhero_retrieval_genres)))
                require_animal_music_combo = bool(require_animal_theme and require_music_theme)
                exact_threshold = 5
                _debug(
                    f"DEBUG parsed genres={parsed.genre_ids} merged genres={list(requested_genres)} query_terms={query_terms}"
                )
                if len(requested_genres) >= 2:
                    exact_discover_params = {
                        "with_genres": ",".join(str(g) for g in requested_genres),
                        "year_min": active_filters.year_min if active_filters.year_active else None,
                        "year_max": active_filters.year_max if active_filters.year_active else None,
                        "min_rating": active_filters.min_rating,
                        "min_vote_count": active_filters.min_vote_count if active_filters.vote_count_active else None,
                        "runtime_min": active_filters.runtime_min if active_filters.enforce_runtime else None,
                        "runtime_max": active_filters.runtime_max if active_filters.enforce_runtime else None,
                        "language": active_filters.language,
                        "pages": 8,
                        "limit": 320,
                        "sort_by": "vote_count.desc",
                    }
                    _debug(f"DEBUG discover exact params={exact_discover_params}")
                    exact_corpus = discover_hydrated(
                        discover_genres,
                        year_min=active_filters.year_min if active_filters.year_active else None,
                        year_max=active_filters.year_max if active_filters.year_active else None,
                        min_rating=active_filters.min_rating,
                        min_vote_count=active_filters.min_vote_count if active_filters.vote_count_active else None,
                        runtime_min=active_filters.runtime_min if active_filters.enforce_runtime else None,
                        runtime_max=active_filters.runtime_max if active_filters.enforce_runtime else None,
                        language=active_filters.language,
                        pages=12 if required_franchise else (10 if require_animal_music_combo else 8),
                        limit=520 if required_franchise else (500 if require_animal_music_combo else 320),
                        genre_joiner="|" if require_superhero else ",",
                        sort_by="vote_count.desc",
                    )
                    if relationship_anchor and not anchor and relationship_anchor.get("id"):
                        related = fetch_anchor_candidate_pool(int(relationship_anchor["id"]), pages=4)
                        exact_corpus = _merge_movies_unique(exact_corpus, related)
                    _debug(f"DEBUG discover exact count={len(exact_corpus)}")
                    exact_corpus = filter_mod.apply_hard_exclusions(
                        exact_corpus, active_filters, debug_context="smart:multi-genre-exact"
                    )
                    _debug(f"DEBUG after hard exclusions={len(exact_corpus)}")
                    filtered = filter_mod.apply_filters(
                        exact_corpus,
                        active_filters,
                        relaxed_genres_for_and=False,
                    )
                    if required_franchise:
                        filtered = [m for m in filtered if filter_mod.matches_franchise(m, str(required_franchise))]
                    if require_superhero:
                        filtered, tier_stats = recommender.apply_superhero_tiered_filter(
                            filtered,
                            require_romance=require_romance,
                        )
                        _debug(
                            "DEBUG superhero tiers "
                            f"retrieval={tier_stats.get('retrieval')} strict={tier_stats.get('strict')} "
                            f"partial={tier_stats.get('partial')} soft={tier_stats.get('soft')} "
                            f"used={tier_stats.get('tier_used')}"
                        )
                        if not filtered and active_filters.studio_contains:
                            # Avoid hard-zero from studio filter in theme-based queries; keep user intent via theme tiers.
                            relaxed = filter_mod.MovieFilters(**{**active_filters.__dict__})
                            relaxed.studio_contains = None
                            filtered = filter_mod.apply_filters(
                                exact_corpus,
                                relaxed,
                                relaxed_genres_for_and=False,
                            )
                            if required_franchise:
                                filtered = [m for m in filtered if filter_mod.matches_franchise(m, str(required_franchise))]
                            filtered, tier_stats = recommender.apply_superhero_tiered_filter(
                                filtered,
                                require_romance=require_romance,
                            )
                            _info_once(
                                "No exact matches for your studio + theme combination. Showing the closest matches instead."
                            )
                    if require_animal_theme or require_music_theme:
                        filtered, theme_stats = recommender.apply_broad_theme_tiered_filter(
                            filtered,
                            require_animal_theme=require_animal_theme,
                            require_music_theme=require_music_theme,
                        )
                        _debug(
                            "DEBUG broad-theme tiers "
                            f"retrieval={theme_stats.get('retrieval')} strict={theme_stats.get('strict')} "
                            f"partial={theme_stats.get('partial')} soft={theme_stats.get('soft')} "
                            f"used={theme_stats.get('tier_used')}"
                        )
                    _debug(f"DEBUG after regular filters={len(filtered)}")
                    filtered = apply_quality_gate(filtered, min_votes=max(80, int(active_filters.min_vote_count or 0)))
                    _debug(f"DEBUG after quality gate={len(filtered)}")
                else:
                    corpus = discover_hydrated(
                        discover_genres,
                        year_min=active_filters.year_min if active_filters.year_active else None,
                        year_max=active_filters.year_max if active_filters.year_active else None,
                        min_rating=active_filters.min_rating,
                        min_vote_count=active_filters.min_vote_count if active_filters.vote_count_active else None,
                        runtime_min=active_filters.runtime_min if active_filters.enforce_runtime else None,
                        runtime_max=active_filters.runtime_max if active_filters.enforce_runtime else None,
                        language=active_filters.language,
                        pages=10 if required_franchise else (8 if require_animal_music_combo else 5),
                        limit=520 if required_franchise else (450 if require_animal_music_combo else 220),
                        genre_joiner="|" if require_superhero else ",",
                        sort_by="popularity.desc",
                    )
                    if relationship_anchor and not anchor and relationship_anchor.get("id"):
                        related = fetch_anchor_candidate_pool(int(relationship_anchor["id"]), pages=4)
                        corpus = _merge_movies_unique(corpus, related)
                    corpus = filter_mod.apply_hard_exclusions(corpus, active_filters, debug_context="smart:initial")
                    filtered = filter_mod.apply_filters(
                        corpus,
                        active_filters,
                        relaxed_genres_for_and=False,
                    )
                    if required_franchise:
                        filtered = [m for m in filtered if filter_mod.matches_franchise(m, str(required_franchise))]
                    if require_superhero:
                        filtered, tier_stats = recommender.apply_superhero_tiered_filter(
                            filtered,
                            require_romance=require_romance,
                        )
                        _debug(
                            "DEBUG superhero tiers "
                            f"retrieval={tier_stats.get('retrieval')} strict={tier_stats.get('strict')} "
                            f"partial={tier_stats.get('partial')} soft={tier_stats.get('soft')} "
                            f"used={tier_stats.get('tier_used')}"
                        )
                        if not filtered and active_filters.studio_contains:
                            relaxed = filter_mod.MovieFilters(**{**active_filters.__dict__})
                            relaxed.studio_contains = None
                            filtered = filter_mod.apply_filters(
                                corpus,
                                relaxed,
                                relaxed_genres_for_and=False,
                            )
                            if required_franchise:
                                filtered = [m for m in filtered if filter_mod.matches_franchise(m, str(required_franchise))]
                            filtered, tier_stats = recommender.apply_superhero_tiered_filter(
                                filtered,
                                require_romance=require_romance,
                            )
                            _info_once(
                                "No exact matches for your studio + theme combination. Showing the closest matches instead."
                            )
                        if not filtered and active_filters.audience_level:
                            # As a last resort, keep safety exclusions but relax audience gating.
                            relaxed = filter_mod.MovieFilters(**{**active_filters.__dict__})
                            relaxed.audience_level = None
                            filtered = filter_mod.apply_filters(
                                corpus,
                                relaxed,
                                relaxed_genres_for_and=False,
                            )
                            if required_franchise:
                                filtered = [m for m in filtered if filter_mod.matches_franchise(m, str(required_franchise))]
                            filtered, tier_stats = recommender.apply_superhero_tiered_filter(
                                filtered,
                                require_romance=require_romance,
                            )
                            if filtered:
                                _info_once(
                                    "No exact matches for your audience level + theme combination. Showing closest matches instead."
                                )
                    if require_animal_theme or require_music_theme:
                        filtered, theme_stats = recommender.apply_broad_theme_tiered_filter(
                            filtered,
                            require_animal_theme=require_animal_theme,
                            require_music_theme=require_music_theme,
                        )
                        _debug(
                            "DEBUG broad-theme tiers "
                            f"retrieval={theme_stats.get('retrieval')} strict={theme_stats.get('strict')} "
                            f"partial={theme_stats.get('partial')} soft={theme_stats.get('soft')} "
                            f"used={theme_stats.get('tier_used')}"
                        )
                    _debug(
                        f"DEBUG discover count={len(corpus)} filtered count={len(filtered)}"
                    )
                if not filtered:
                    st.warning("No results match your filters. Try relaxing your constraints.")
                    _debug("DEBUG zero cause: filtered list empty before ranking.")
                    ranked: list[tuple[dict[str, Any], float, dict[str, float]]] = []
                else:
                    # 4) similarity/ranking
                    exact_count = 0
                    if anchor:
                        exact_ranked, used_fallback = recommender.hybrid_with_fallback(
                            anchor,
                            filtered,
                            top_n=30,
                            mood_sentiment_target=mood_tgt,
                            relationship_context=relationship_context,
                        )
                        closest_ranked = recommender.hybrid_recommendations(
                            anchor,
                            filtered,
                            top_n=30,
                            mood_sentiment_target=mood_tgt,
                            content_sim_threshold=0.05,
                            relationship_context=relationship_context,
                        )
                    else:
                        # Query-only OR genre-only
                        qtext = " ".join(query_terms) if query_terms else ""
                        query_intent = nlp_utils.parse_query_intent(nl or "", parsed) if qtext else None
                        structured_genre_query = nlp_utils.is_short_structured_genre_query(
                            nl or "",
                            parsed,
                            query_terms,
                        )
                        if qtext:
                            ranked_general = recommender.rank_candidates(
                                filtered,
                                query_intent if query_intent is not None else nlp_utils.parse_query_intent(nl or "", parsed),
                                top_n=60,
                                anchor=relationship_anchor,
                                relationship_context=relationship_context,
                            )
                            ex_tier, close_tier, broad_tier = recommender.split_ranked_tiers(ranked_general)
                            exact_ranked = ex_tier
                            closest_ranked = close_tier + broad_tier
                            used_fallback = bool(not ex_tier and (close_tier or broad_tier))
                            if not ranked_general and len(filtered) > 0:
                                _debug(
                                    "DEBUG zero cause: general scorer returned no candidates; "
                                    "switching to genre-only ranking fallback."
                                )
                                exact_ranked = recommender.genre_only_recommendations(
                                    filtered,
                                    top_n=30,
                                    requested_genre_ids=set(active_filters.genre_ids) if active_filters.genre_ids else None,
                                    keyword_hint=active_filters.keyword_contains,
                                )
                                closest_ranked = []
                                used_fallback = False
                        elif structured_genre_query:
                            _debug("DEBUG short structured genre query detected; using genre-first ranking.")
                            exact_ranked = recommender.genre_only_recommendations(
                                filtered,
                                top_n=30,
                                requested_genre_ids=set(active_filters.genre_ids) if active_filters.genre_ids else None,
                                keyword_hint=active_filters.keyword_contains,
                            )
                            closest_ranked = []
                            used_fallback = False
                        else:
                            exact_ranked = recommender.genre_only_recommendations(
                                filtered,
                                top_n=30,
                                requested_genre_ids=set(active_filters.genre_ids) if active_filters.genre_ids else None,
                                keyword_hint=active_filters.keyword_contains,
                            )
                            used_fallback = False
                            closest_ranked = []
                    exact_count = len(exact_ranked)
                    ranked = recommender.build_result_list(exact_ranked, closest_ranked, target_n=target_n)
                    _debug(
                        f"DEBUG ranked counts exact={len(exact_ranked)} closest={len(closest_ranked)} final={len(ranked)}"
                    )
                    if exact_count > 0:
                        if len(ranked) > exact_count:
                            _info_once("Showing exact matches first, followed by closest additional matches.")
                        else:
                            _info_once("Showing exact matches for your genres and filters.")
                    elif used_fallback and ranked:
                        _info_once("No exact matches found. Showing the closest matches instead.")
                    if safe_compromise and ranked:
                        _info_once("Showing the closest family-safe matches based on your constraints.")

                    # Family-safe conflict fallback:
                    # If Kids/Family + mature-adjacent genres yields sparse exact results, switch
                    # retrieval toward family-safe adjacent genres (while keeping safety hard).
                    if family_conflict_fallback and len(ranked) < 3:
                        fallback_filters = filter_mod.MovieFilters(
                            year_min=active_filters.year_min,
                            year_max=active_filters.year_max,
                            min_rating=active_filters.min_rating,
                            min_vote_count=active_filters.min_vote_count,
                            runtime_min=active_filters.runtime_min,
                            runtime_max=active_filters.runtime_max,
                            certification=active_filters.certification,
                            language=active_filters.language,
                            actor_names=list(active_filters.actor_names),
                            director_names=list(active_filters.director_names),
                            genre_ids=build_family_safe_adjacent_genres(active_filters.genre_ids),
                            genre_mode="OR",
                            enforce_runtime=active_filters.enforce_runtime,
                            year_active=active_filters.year_active,
                            vote_count_active=active_filters.vote_count_active,
                            studio_contains=active_filters.studio_contains,
                            country_contains=active_filters.country_contains,
                            keyword_contains=active_filters.keyword_contains,
                            exclude_animation=active_filters.exclude_animation,
                            exclude_documentary=active_filters.exclude_documentary,
                            exclude_horror=active_filters.exclude_horror,
                            exclude_adult=active_filters.exclude_adult,
                                exclude_superhero=active_filters.exclude_superhero,
                                excluded_genre_ids=list(active_filters.excluded_genre_ids),
                            audience_level=active_filters.audience_level,
                            sort_by=active_filters.sort_by,
                            allow_uncertified_in_fallback=True,
                        )

                        family_more = discover_hydrated(
                            tuple(sorted(fallback_filters.genre_ids)),
                            year_min=fallback_filters.year_min if fallback_filters.year_active else None,
                            year_max=fallback_filters.year_max if fallback_filters.year_active else None,
                            min_rating=fallback_filters.min_rating,
                            min_vote_count=fallback_filters.min_vote_count if fallback_filters.vote_count_active else None,
                            runtime_min=fallback_filters.runtime_min if fallback_filters.enforce_runtime else None,
                            runtime_max=fallback_filters.runtime_max if fallback_filters.enforce_runtime else None,
                            language=fallback_filters.language,
                            pages=8,
                            limit=320,
                        )
                        family_more = filter_mod.apply_hard_exclusions(
                            family_more,
                            fallback_filters,
                            debug_context="smart:family-conflict-fallback",
                        )
                        family_filtered = filter_mod.apply_filters(
                            family_more,
                            fallback_filters,
                            relaxed_genres_for_and=True,
                        )
                        if anchor:
                            fam_exact, _ = recommender.hybrid_with_fallback(
                                anchor,
                                family_filtered,
                                top_n=30,
                                mood_sentiment_target=mood_tgt,
                            )
                            fam_close = recommender.hybrid_recommendations(
                                anchor,
                                family_filtered,
                                top_n=30,
                                mood_sentiment_target=mood_tgt,
                                content_sim_threshold=0.05,
                            )
                        else:
                            qtext_family = " ".join(query_terms) if query_terms else "family mystery adventure comedy detective caper"
                            fam_exact, _ = recommender.query_with_fallback(
                                qtext_family,
                                family_filtered,
                                top_n=30,
                                mood_sentiment_target=mood_tgt,
                                requested_genre_ids=set(fallback_filters.genre_ids),
                            )
                            fam_close = recommender.query_recommendations(
                                qtext_family,
                                family_filtered,
                                top_n=30,
                                mood_sentiment_target=mood_tgt,
                                requested_genre_ids=set(fallback_filters.genre_ids),
                                content_sim_threshold=0.05,
                            )
                        family_ranked = recommender.build_result_list(fam_exact, fam_close, target_n=target_n)
                        ranked = recommender.build_result_list(ranked, family_ranked, target_n=target_n)
                        if family_ranked:
                            _info_once(
                                "No exact Kids / Family crime matches were found. "
                                "Showing the closest family-safe crime-adjacent titles instead."
                            )
                            safe_compromise = True

                    if len(ranked) < target_n:
                        # Fallback step: fetch more pages and retry / fill list.
                        # For multi-genre queries, fallback should use OR-genre retrieval, not the same exact pool.
                        more = discover_hydrated(
                            tuple(sorted(active_filters.genre_ids)),
                            year_min=active_filters.year_min if active_filters.year_active else None,
                            year_max=active_filters.year_max if active_filters.year_active else None,
                            min_rating=active_filters.min_rating,
                            min_vote_count=active_filters.min_vote_count if active_filters.vote_count_active else None,
                            runtime_min=active_filters.runtime_min if active_filters.enforce_runtime else None,
                            runtime_max=active_filters.runtime_max if active_filters.enforce_runtime else None,
                            language=active_filters.language,
                            pages=14 if required_franchise else (12 if require_animal_music_combo else 10),
                            limit=650 if required_franchise else (620 if require_animal_music_combo else 350),
                            genre_joiner="|" if len(requested_genres) >= 2 else ",",
                            sort_by="popularity.desc",
                        )
                        _debug(f"DEBUG fallback discover count={len(more)}")
                        more = filter_mod.apply_hard_exclusions(more, active_filters, debug_context="smart:fallback")
                        _debug(f"DEBUG fallback after hard exclusions={len(more)}")
                        filtered_more = filter_mod.apply_filters(
                            more,
                            active_filters,
                            relaxed_genres_for_and=True,
                        )
                        if required_franchise:
                            filtered_more = [m for m in filtered_more if filter_mod.matches_franchise(m, str(required_franchise))]
                        if require_superhero:
                            filtered_more, tier_stats = recommender.apply_superhero_tiered_filter(
                                filtered_more,
                                require_romance=require_romance,
                            )
                            _debug(
                                "DEBUG superhero tiers fallback "
                                f"retrieval={tier_stats.get('retrieval')} strict={tier_stats.get('strict')} "
                                f"partial={tier_stats.get('partial')} soft={tier_stats.get('soft')} "
                                f"used={tier_stats.get('tier_used')}"
                            )
                            if not filtered_more and active_filters.studio_contains:
                                relaxed = filter_mod.MovieFilters(**{**active_filters.__dict__})
                                relaxed.studio_contains = None
                                filtered_more = filter_mod.apply_filters(
                                    more,
                                    relaxed,
                                    relaxed_genres_for_and=True,
                                )
                                if required_franchise:
                                    filtered_more = [m for m in filtered_more if filter_mod.matches_franchise(m, str(required_franchise))]
                                filtered_more, tier_stats = recommender.apply_superhero_tiered_filter(
                                    filtered_more,
                                    require_romance=require_romance,
                                )
                                if filtered_more:
                                    _info_once(
                                        "No exact matches for your studio + theme combination. Showing the closest matches instead."
                                    )
                        if require_animal_theme or require_music_theme:
                            filtered_more, theme_stats = recommender.apply_broad_theme_tiered_filter(
                                filtered_more,
                                require_animal_theme=require_animal_theme,
                                require_music_theme=require_music_theme,
                            )
                            _debug(
                                "DEBUG broad-theme tiers fallback "
                                f"retrieval={theme_stats.get('retrieval')} strict={theme_stats.get('strict')} "
                                f"partial={theme_stats.get('partial')} soft={theme_stats.get('soft')} "
                                f"used={theme_stats.get('tier_used')}"
                            )
                        _debug(f"DEBUG fallback after regular filters={len(filtered_more)}")
                        filtered_more = apply_quality_gate(
                            filtered_more, min_votes=max(40, int(active_filters.min_vote_count or 0))
                        )
                        _debug(f"DEBUG fallback after quality gate={len(filtered_more)}")
                        if anchor:
                            exact_ranked, used_fallback = recommender.hybrid_with_fallback(
                                anchor,
                                filtered_more,
                                top_n=30,
                                mood_sentiment_target=mood_tgt,
                                relationship_context=relationship_context,
                            )
                            closest_ranked = recommender.hybrid_recommendations(
                                anchor,
                                filtered_more,
                                top_n=30,
                                mood_sentiment_target=mood_tgt,
                                content_sim_threshold=0.05,
                                relationship_context=relationship_context,
                            )
                        elif qtext:
                            ranked_general_fb = recommender.rank_candidates(
                                filtered_more,
                                query_intent if query_intent is not None else nlp_utils.parse_query_intent(nl or "", parsed),
                                top_n=60,
                                anchor=relationship_anchor,
                                relationship_context=relationship_context,
                            )
                            ex_tier_fb, close_tier_fb, broad_tier_fb = recommender.split_ranked_tiers(ranked_general_fb)
                            exact_ranked = ex_tier_fb
                            closest_ranked = close_tier_fb + broad_tier_fb
                            used_fallback = bool(not ex_tier_fb and (close_tier_fb or broad_tier_fb))
                        else:
                            exact_ranked = recommender.genre_only_recommendations(
                                filtered_more,
                                top_n=30,
                                requested_genre_ids=set(active_filters.genre_ids) if active_filters.genre_ids else None,
                                keyword_hint=active_filters.keyword_contains,
                            )
                            used_fallback = True if exact_ranked else False
                            closest_ranked = []
                        # Merge existing ranked with broader fallback-ranked pool.
                        expanded_ranked = recommender.build_result_list(exact_ranked, closest_ranked, target_n=target_n * 3)
                        ranked = recommender.build_result_list(ranked, expanded_ranked, target_n=target_n)
                        _debug(f"DEBUG fallback ranked final={len(ranked)}")

                        if len(ranked) >= 1:
                            if exact_count > 0:
                                if len(ranked) > exact_count:
                                    _info_once("Showing exact matches first, followed by closest additional matches.")
                                else:
                                    _info_once("Showing exact matches for your genres and filters.")
                            else:
                                _info_once("No exact matches found. Showing the closest matches instead.")
                        else:
                            st.warning("No results match your filters. Try relaxing your constraints.")
                    ranked = sort_ranked_results(ranked, active_filters.sort_by)
                    if require_superhero:
                        ranked = [
                            (m, s, {**c, "required_superhero_theme": 1.0 if recommender.is_superhero_movie(m) else 0.0})
                            for (m, s, c) in ranked
                        ]
                    if required_franchise:
                        ranked = [(m, s, {**c, "required_franchise": str(required_franchise)}) for (m, s, c) in ranked]
                    if require_animal_theme:
                        ranked = [
                            (m, s, {**c, "required_animal_theme": 1.0 if recommender.score_animal_theme(m) >= 1 else 0.0})
                            for (m, s, c) in ranked
                        ]
                    if require_music_theme:
                        ranked = [
                            (m, s, {**c, "required_music_theme": 1.0 if recommender.score_music_theme(m) >= 1 else 0.0})
                            for (m, s, c) in ranked
                        ]
                    if require_animal_theme or require_music_theme:
                        # Improve best-match ordering for broad multi-theme queries.
                        boosted: list[tuple[dict[str, Any], float, dict[str, float]]] = []
                        max_votes_hint = (
                            float(max((m.get("vote_count") or 0) for (m, _s, _c) in ranked) or 1.0) if ranked else 1.0
                        )
                        for m, s, c in ranked:
                            a_sc = float(recommender.score_animal_theme(m))
                            mu_sc = float(recommender.score_music_theme(m))
                            combined = a_sc + mu_sc
                            min_pair = min(a_sc, mu_sc) if (require_animal_theme and require_music_theme) else 0.0
                            pop_sc = float(recommender.popularity_confidence_score(m, max_votes_hint=max_votes_hint))
                            canon_boost = float(
                                recommender.canonical_theme_boost(
                                    m,
                                    query_text=(nl or ""),
                                    require_animal_theme=require_animal_theme,
                                    require_music_theme=require_music_theme,
                                )
                            )
                            semantic = max(0.0, min(1.0, float(s) / 1.2))
                            theme_norm = max(0.0, min(1.0, combined / 16.0))
                            quality = (0.6 * semantic) + (0.2 * theme_norm) + (0.2 * pop_sc) + (0.20 * canon_boost)
                            strength = "strong" if (a_sc >= 4 and mu_sc >= 4 and require_animal_theme and require_music_theme) else "partial"
                            boosted.append(
                                (
                                    m,
                                    float(quality),
                                    {
                                        **c,
                                        "animal_theme_score": a_sc,
                                        "music_theme_score": mu_sc,
                                        "multi_theme_score": float(combined),
                                        "multi_theme_strength": strength,
                                        "canonical_theme_boost": float(canon_boost),
                                        "theme_quality_score": float(quality),
                                    },
                                )
                            )
                        ranked = sorted(
                            boosted,
                            key=lambda x: (
                                float(x[2].get("canonical_theme_boost", 0.0)),
                                float(x[2].get("multi_theme_score", 0.0)),
                                float(min(x[2].get("animal_theme_score", 0.0), x[2].get("music_theme_score", 0.0)))
                                if (require_animal_theme and require_music_theme)
                                else 0.0,
                                float(x[2].get("theme_quality_score", x[1])),
                            ),
                            reverse=True,
                        )
                    if safe_compromise:
                        ranked = [(m, s, {**c, "match_level": "closest_safe"}) for (m, s, c) in ranked]
                    # Final safety net before render/explanations.
                    ranked = [
                        (m, s, c)
                        for (m, s, c) in ranked
                        if not filter_mod.violates_exclusions(m, active_filters)
                    ]

                st.session_state["smart_ranked"] = (anchor, ranked)

    stored = st.session_state.get("smart_ranked")
    if not stored:
        save_prefix_to_shared("smart")
        return
    anch, ranked = stored
    if anch:
        st.success(f"Anchor: **{anch.get('title')}** — showing blended matches.")
    else:
        st.success("Showing matches from your query + filters.")
    if not ranked:
        st.info("No recommendations to display. Generate again after adjusting filters.")
        save_prefix_to_shared("smart")
        return
    for movie, score, comps in ranked:
        with st.container():
            c1, c2 = st.columns([1, 3])
            with c1:
                ui_components.movie_card(movie)
                if st.button("Watchlist +", key=f"wl_smart_{movie.get('id')}"):
                    add_to_watchlist(movie)
                    st.success("Added to watchlist.")
            with c2:
                st.markdown(f"**Score** `{score:.3f}`  \n**Why this title:**")
                ui_components.explanation_block(
                    recommender.explanation_blurb(anch, movie, comps),
                )
        st.divider()
    save_prefix_to_shared("smart")


def page_similar(genre_options: list[dict[str, Any]]) -> None:
    st.header("Similar movies")
    st.caption("Exact similar matches are prioritized first. If matches are sparse, closest genre-overlap matches are shown.")
    filt_col, _ = st.columns([1, 10])
    with filt_col:
        render_movie_filters_panel(
            genre_options,
            "sim",
            visibility_key="show_similar_filters",
            toggle_button_key="similar_filters_btn",
            title="Similar Movies Filters",
        )
    base_filters = movie_filters_from_prefix("sim", genre_options)
    q = st.text_input("Find a movie to match", placeholder="Blade Runner 2049", key="similar_anchor_query")
    if st.button(
        "Find titles",
        key="similar_find_titles_btn",
        on_click=_close_modal_flag,
        args=("show_similar_filters",),
    ):
        qn_new = (q or "").strip()
        st.session_state["sim_q"] = qn_new
        if qn_new:
            res = api.search_movies(qn_new, page=1)
            rows: list[dict[str, Any]] = []
            label_to_id: dict[str, int] = {}
            for r in res[:20]:
                mid = r.get("id")
                if not isinstance(mid, int):
                    continue
                label = f"{r.get('title', 'Untitled')} ({str(r.get('release_date') or '')[:4]})"
                rows.append({"id": int(mid), "label": label})
                label_to_id[label] = int(mid)
            st.session_state["similar_search_results"] = rows
            st.session_state["similar_search_label_to_id"] = label_to_id
            if rows:
                default_label = rows[0]["label"]
                st.session_state["similar_anchor_select"] = default_label
                st.session_state["similar_selected_movie_id"] = int(rows[0]["id"])
            else:
                st.session_state["similar_selected_movie_id"] = None
        else:
            st.session_state["similar_search_results"] = []
            st.session_state["similar_search_label_to_id"] = {}
            st.session_state["similar_selected_movie_id"] = None

    qn = st.session_state.get("sim_q", "")
    anchor: dict[str, Any] | None = None
    if qn:
        rows = st.session_state.get("similar_search_results", [])
        label_to_id = st.session_state.get("similar_search_label_to_id", {})
        if not rows:
            # Lazy initialize once if query existed before mapping state was introduced.
            res = api.search_movies(qn, page=1)
            rows = []
            label_to_id = {}
            for r in res[:20]:
                mid = r.get("id")
                if not isinstance(mid, int):
                    continue
                label = f"{r.get('title', 'Untitled')} ({str(r.get('release_date') or '')[:4]})"
                rows.append({"id": int(mid), "label": label})
                label_to_id[label] = int(mid)
            st.session_state["similar_search_results"] = rows
            st.session_state["similar_search_label_to_id"] = label_to_id

        if not rows:
            st.warning("No results.")
        else:
            labels = [str(r.get("label") or "") for r in rows if r.get("label")]
            if labels and st.session_state.get("similar_anchor_select") not in labels:
                st.session_state["similar_anchor_select"] = labels[0]
            pick = st.selectbox(
                "Choose anchor movie",
                labels,
                key="similar_anchor_select",
                on_change=_on_similar_anchor_change,
            )
            selected_id = st.session_state.get("similar_selected_movie_id")
            if not isinstance(selected_id, int):
                raw_id = label_to_id.get(pick)
                if isinstance(raw_id, int):
                    selected_id = int(raw_id)
                    st.session_state["similar_selected_movie_id"] = selected_id
            if isinstance(selected_id, int):
                anchor = api.get_movie_details(int(selected_id))

    selected_movie_id = st.session_state.get("similar_selected_movie_id")
    if not isinstance(selected_movie_id, int):
        st.info("Search and select a movie to see similar picks.")
        save_prefix_to_shared("sim")
        return

    # Keep anchor details stable across reruns by caching by selected movie id.
    anchor = api.get_movie_details(int(selected_movie_id))
    if not anchor:
        st.warning("Could not load the selected movie details. Please pick another title and try again.")
        save_prefix_to_shared("sim")
        return

    relationship_context = build_anchor_relationship_context(int(selected_movie_id), pages=4)
    target_n = 10
    sim_sig = (
        int(selected_movie_id),
        tuple(sorted(base_filters.genre_ids)),
        base_filters.genre_mode,
        base_filters.year_min,
        base_filters.year_max,
        base_filters.year_active,
        base_filters.min_rating,
        base_filters.min_vote_count,
        base_filters.vote_count_active,
        base_filters.runtime_min,
        base_filters.runtime_max,
        base_filters.enforce_runtime,
        base_filters.certification,
        base_filters.audience_level,
        base_filters.language,
        base_filters.studio_contains,
        base_filters.country_contains,
        base_filters.keyword_contains,
        base_filters.exclude_animation,
        base_filters.exclude_documentary,
        base_filters.exclude_horror,
        base_filters.exclude_adult,
    )
    if st.session_state.get("sim_filter_sig") != sim_sig:
        st.session_state.pop("sim_ranked", None)
        st.session_state["sim_filter_sig"] = sim_sig

    manual_compute = st.button(
        "Compute similar",
        type="primary",
        key="similar_compute_btn",
        on_click=_close_modal_flag,
        args=("show_similar_filters",),
    )
    auto_compute = bool(st.session_state.pop("sim_auto_compute", False))
    should_compute = bool(selected_movie_id) and (manual_compute or auto_compute)
    if manual_compute and not isinstance(selected_movie_id, int):
        st.warning("Please select a movie from the search results first.")
        save_prefix_to_shared("sim")
        return
    if manual_compute and isinstance(selected_movie_id, int) and not anchor:
        st.warning("Could not load the selected movie details. Please pick another title and try again.")
        save_prefix_to_shared("sim")
        return
    if should_compute:
        with st.spinner("Scoring similarity…"):
            # Dedicated anchor-first pool (similar + recommendations), not generic trending cache.
            corpus = fetch_anchor_candidate_pool(int(selected_movie_id), pages=4)
            if not corpus:
                st.warning("Could not load anchor-related candidates from TMDB.")
                st.session_state["sim_ranked"] = (anchor, [])
                save_prefix_to_shared("sim")
                return
            corpus = filter_mod.apply_hard_exclusions(corpus, base_filters, debug_context="similar:initial")

            if base_filters.enforce_runtime and base_filters.vote_count_active:
                st.warning(
                    "Your current runtime and vote-count filters are very restrictive for this anchor. "
                    "Showing broader matches may help."
                )

            filtered = filter_mod.apply_filters(
                corpus,
                base_filters,
                relaxed_genres_for_and=True,
            )
            exact_ranked, used_fallback = recommender.hybrid_with_fallback(
                anchor,
                filtered,
                top_n=30,
                mood_sentiment_target=None,
                content_sim_threshold=0.15,
                fallback_content_sim_threshold=0.10,
                w_content=0.40,
                w_genre=0.20,
                w_sentiment=0.10,
                w_popularity=0.10,
                relationship_context=relationship_context,
            )
            closest_ranked = recommender.hybrid_recommendations(
                anchor,
                filtered,
                top_n=30,
                mood_sentiment_target=None,
                content_sim_threshold=0.05,
                w_content=0.40,
                w_genre=0.20,
                w_sentiment=0.10,
                w_popularity=0.10,
                relationship_context=relationship_context,
            )
            ranked = recommender.build_result_list(exact_ranked, closest_ranked, target_n=target_n)
            if len(ranked) < target_n:
                # Fallback 1: relax runtime and vote-count hard filters
                fallback_filters = filter_mod.MovieFilters(
                    year_min=base_filters.year_min,
                    year_max=base_filters.year_max,
                    min_rating=base_filters.min_rating,
                    min_vote_count=base_filters.min_vote_count,
                    runtime_min=base_filters.runtime_min,
                    runtime_max=base_filters.runtime_max,
                    certification=base_filters.certification,
                    language=base_filters.language,
                    actor_names=list(base_filters.actor_names),
                    director_names=list(base_filters.director_names),
                    genre_ids=list(base_filters.genre_ids),
                    genre_mode=base_filters.genre_mode,
                    year_active=base_filters.year_active,
                    vote_count_active=False,
                    enforce_runtime=False,
                    studio_contains=base_filters.studio_contains,
                    country_contains=base_filters.country_contains,
                    keyword_contains=base_filters.keyword_contains,
                    exclude_animation=base_filters.exclude_animation,
                    exclude_documentary=base_filters.exclude_documentary,
                    exclude_horror=base_filters.exclude_horror,
                    exclude_adult=base_filters.exclude_adult,
                    audience_level=base_filters.audience_level,
                    sort_by=base_filters.sort_by,
                    allow_uncertified_in_fallback=True,
                )
                fallback_filtered = filter_mod.apply_filters(
                    corpus,
                    fallback_filters,
                    relaxed_genres_for_and=True,
                )
                # Broader closest pool: anchor genres via discover, not only same filtered list.
                anchor_genres = [g.get("id") for g in (anchor.get("genres") or []) if isinstance(g, dict) and g.get("id")]
                broadened = discover_hydrated(
                    tuple(sorted(int(g) for g in anchor_genres if g)),
                    year_min=None,
                    year_max=None,
                    min_rating=fallback_filters.min_rating,
                    min_vote_count=None,
                    runtime_min=None,
                    runtime_max=None,
                    language=fallback_filters.language,
                    pages=6,
                    limit=260,
                )
                broadened = filter_mod.apply_hard_exclusions(broadened, fallback_filters, debug_context="similar:fallback-broadened")
                fallback_filtered = filter_mod.apply_filters(
                    list({m.get("id"): m for m in (fallback_filtered + broadened) if m.get("id")}.values()),
                    fallback_filters,
                    relaxed_genres_for_and=True,
                )
                exact_ranked, used_fallback = recommender.hybrid_with_fallback(
                    anchor,
                    fallback_filtered,
                    top_n=30,
                    mood_sentiment_target=None,
                    content_sim_threshold=0.15,
                    fallback_content_sim_threshold=0.10,
                    w_content=0.40,
                    w_genre=0.20,
                    w_sentiment=0.10,
                    w_popularity=0.10,
                    relationship_context=relationship_context,
                )
                closest_ranked = recommender.hybrid_recommendations(
                    anchor,
                    fallback_filtered,
                    top_n=30,
                    mood_sentiment_target=None,
                    content_sim_threshold=0.05,
                    w_content=0.40,
                    w_genre=0.20,
                    w_sentiment=0.10,
                    w_popularity=0.10,
                    relationship_context=relationship_context,
                )
                expanded_ranked = recommender.build_result_list(exact_ranked, closest_ranked, target_n=target_n * 3)
                ranked = recommender.build_result_list(ranked, expanded_ranked, target_n=target_n)
                if ranked:
                    st.info("No exact similar titles found. Showing the closest matches instead.")
                else:
                    st.warning("No similar titles after filters — try wider filters.")
            elif used_fallback:
                st.info("No exact similar titles found. Showing the closest genre-overlap matches.")
            ranked = sort_ranked_results(ranked, base_filters.sort_by)
            # Final safety net before render/explanations.
            ranked = [
                (m, s, c)
                for (m, s, c) in ranked
                if not filter_mod.violates_exclusions(m, base_filters)
            ]
        st.session_state["sim_ranked"] = (anchor, ranked)

    pack = st.session_state.get("sim_ranked")
    if not pack:
        save_prefix_to_shared("sim")
        return
    an, ranked = pack
    if not ranked:
        st.info("No similar movies to display.")
        save_prefix_to_shared("sim")
        return
    for movie, score, comps in ranked:
        c1, c2 = st.columns([1, 3])
        with c1:
            ui_components.movie_card(movie)
        with c2:
            st.metric("Hybrid similarity", f"{score:.3f}")
            ui_components.explanation_block(recommender.explanation_blurb(an, movie, comps))
        st.divider()
    save_prefix_to_shared("sim")


def page_watchlist() -> None:
    st.header("Watchlist")
    _ensure_watchlist_state()
    wl = st.session_state.watchlist
    if not wl:
        st.info("Your watchlist is empty. Add titles from Smart recommendations.")
        return
    for row_start in range(0, len(wl), 4):
        row = wl[row_start : row_start + 4]
        cols = st.columns(4)
        for col, m in zip(cols, row):
            with col:
                ui_components.movie_card(m)
                if st.button("Remove", key=f"rm_{m['id']}"):
                    remove_from_watchlist(int(m["id"]))
                    st.rerun()


def page_insights(genre_options: list[dict[str, Any]]) -> None:
    st.header("Insights")
    st.caption("Distributions from the hydrated discovery corpus after your current filters.")
    base_filters = movie_filters_from_shared_dict(genre_options)
    with st.spinner("Preparing charts…"):
        corpus = load_candidate_corpus()
        filtered = filter_mod.apply_filters(corpus, base_filters)
        # Optional enrichment: runtime is only available on full movie details.
        # This keeps Insights resilient if future candidate pools come from endpoints without runtime.
        missing_runtime = [m for m in filtered if m.get("runtime") in (None, 0) and m.get("id")]
        if missing_runtime:
            for m in missing_runtime[:20]:
                d = api.get_movie_details(int(m["id"]))
                if d and d.get("runtime"):
                    m["runtime"] = d.get("runtime")
    if len(filtered) < 5:
        st.warning("Not enough filtered movies for rich charts — relax filters.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(visuals.ratings_distribution_fig(filtered), use_container_width=True)
    with c2:
        st.plotly_chart(visuals.genre_frequency_fig(filtered), use_container_width=True)
    st.plotly_chart(visuals.runtime_distribution_fig(filtered), use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="PickAFlick",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_custom_theme_css()
    try_api_or_stop()

    try:
        genre_options = api.get_movie_genres_list()
    except RuntimeError as e:
        ui_components.error_bubble(str(e))
        st.stop()

    with st.sidebar:
        st.markdown("### PickAFlick")
        st.caption("Use **Filters** on Smart recommendations or Similar movies to refine results.")

    tabs = st.tabs(["Home", "Smart recommendations", "Similar movies", "Watchlist", "Insights"])
    with tabs[0]:
        page_home()
    with tabs[1]:
        page_smart(genre_options)
    with tabs[2]:
        page_similar(genre_options)
    with tabs[3]:
        page_watchlist()
    with tabs[4]:
        page_insights(genre_options)


if __name__ == "__main__":
    main()
