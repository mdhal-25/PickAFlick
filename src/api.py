"""
TMDB API v3 client.

Supports either:
- API Read Access Token: sent as Authorization: Bearer <token> (JWT, typically starts with eyJ)
- v3 API key: sent as ?api_key=<key> (short hex string from TMDB dashboard)

All network helpers are wrapped with st.cache_data to limit redundant calls.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def _token() -> str:
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "TMDB_API_KEY is not set. Add your TMDB v3 API key or Read Access Token (see .env.example)."
        )
    return key


def _use_bearer_auth(token: str) -> bool:
    """JWT read tokens start with eyJ; v3 API keys are short hex strings."""
    return token.startswith("eyJ")


def _headers() -> dict[str, str]:
    t = _token()
    h = {"Accept": "application/json"}
    if _use_bearer_auth(t):
        h["Authorization"] = f"Bearer {t}"
    return h


def _auth_params() -> dict[str, str]:
    t = _token()
    if _use_bearer_auth(t):
        return {}
    return {"api_key": t}


def _get(url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | list | None:
    merged: dict[str, Any] = dict(params or {})
    merged.update(_auth_params())
    try:
        r = requests.get(url, headers=_headers(), params=merged, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, (dict, list)) else None
    except requests.RequestException as exc:
        if os.environ.get("PICKAFLICK_DEBUG_API", "").strip() == "1":
            print(f"[TMDB] Request failed: {url} :: {exc}")
        return None


def _results_list(payload: dict[str, Any] | list | None) -> list[dict[str, Any]]:
    """Normalize TMDB page payloads to a list of movie dicts."""
    if isinstance(payload, dict):
        rows = payload.get("results", [])
        return rows if isinstance(rows, list) else []
    if isinstance(payload, list):
        return payload
    return []


def poster_url(path: str | None, size: str = "w500") -> str | None:
    if not path:
        return None
    return f"https://image.tmdb.org/t/p/{size}{path}"


@st.cache_data(ttl=3600, show_spinner=False)
def search_movies(query: str, page: int = 1) -> list[dict[str, Any]]:
    """Search movies by title."""
    if not query or not query.strip():
        return []
    url = f"{TMDB_BASE}/search/movie"
    data = _get(url, {"query": query.strip(), "page": page})
    return _results_list(data)


@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_details(
    movie_id: int,
    *,
    append: str = "credits,keywords,release_dates",
) -> dict[str, Any] | None:
    """Full movie record with appended credits, keywords, and release dates."""
    url = f"{TMDB_BASE}/movie/{int(movie_id)}"
    data = _get(url, {"append_to_response": append})
    return data if isinstance(data, dict) else None


@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_credits(movie_id: int) -> dict[str, Any] | None:
    """Cast and crew for a movie."""
    url = f"{TMDB_BASE}/movie/{int(movie_id)}/credits"
    data = _get(url)
    return data if isinstance(data, dict) else None


@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_keywords(movie_id: int) -> dict[str, Any] | None:
    """TMDB keywords attached to a movie."""
    url = f"{TMDB_BASE}/movie/{int(movie_id)}/keywords"
    data = _get(url)
    return data if isinstance(data, dict) else None


@st.cache_data(ttl=1800, show_spinner=False)
def get_trending_movies(time_window: str = "week") -> list[dict[str, Any]]:
    """Trending movies: day | week."""
    tw = time_window if time_window in ("day", "week") else "week"
    url = f"{TMDB_BASE}/trending/movie/{tw}"
    data = _get(url)
    return _results_list(data)


@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_genres_list() -> list[dict[str, Any]]:
    """All TMDB movie genres (id + name)."""
    url = f"{TMDB_BASE}/genre/movie/list"
    data = _get(url)
    if isinstance(data, dict) and isinstance(data.get("genres"), list):
        return data["genres"]
    return []


@st.cache_data(ttl=1800, show_spinner=False)
def get_popular_movies(page: int = 1) -> list[dict[str, Any]]:
    """Popular movies page."""
    url = f"{TMDB_BASE}/movie/popular"
    data = _get(url, {"page": max(1, int(page))})
    return _results_list(data)


@st.cache_data(ttl=86400, show_spinner=False)
def discover_movies_page(
    page: int = 1,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Generic discover endpoint for larger candidate pools.
    kwargs map to TMDB discover parameters (e.g. sort_by, year gte/lte).
    """
    url = f"{TMDB_BASE}/discover/movie"
    params: dict[str, Any] = {"page": max(1, int(page))}
    for k, v in kwargs.items():
        if v is not None and v != "":
            # TMDB uses snake_case param names
            params[k] = v
    data = _get(url, params)
    return _results_list(data)


@st.cache_data(ttl=3600, show_spinner=False)
def get_similar_movies(movie_id: int, page: int = 1) -> list[dict[str, Any]]:
    """Anchor-specific similar titles."""
    url = f"{TMDB_BASE}/movie/{int(movie_id)}/similar"
    data = _get(url, {"page": max(1, int(page))})
    return _results_list(data)


@st.cache_data(ttl=3600, show_spinner=False)
def get_movie_recommendations(movie_id: int, page: int = 1) -> list[dict[str, Any]]:
    """Anchor-specific TMDB recommendation titles."""
    url = f"{TMDB_BASE}/movie/{int(movie_id)}/recommendations"
    data = _get(url, {"page": max(1, int(page))})
    return _results_list(data)


def get_similar(movie_id: int, page: int = 1) -> list[dict[str, Any]]:
    """Compatibility alias."""
    return get_similar_movies(movie_id, page)


def get_recommendations(movie_id: int, page: int = 1) -> list[dict[str, Any]]:
    """Compatibility alias."""
    return get_movie_recommendations(movie_id, page)


def discover_movies(page: int = 1, **kwargs: Any) -> list[dict[str, Any]]:
    """Compatibility alias."""
    return discover_movies_page(page, **kwargs)


@st.cache_data(ttl=86400, show_spinner=False)
def get_collection_details(collection_id: int) -> dict[str, Any] | None:
    """Collection details with parts list (franchise members)."""
    url = f"{TMDB_BASE}/collection/{int(collection_id)}"
    data = _get(url)
    return data if isinstance(data, dict) else None


def extract_directors(credits: dict[str, Any] | None) -> list[str]:
    if not credits or "crew" not in credits:
        return []
    return [
        c.get("name", "")
        for c in credits.get("crew", [])
        if c.get("job") == "Director" and c.get("name")
    ]


def extract_top_cast(credits: dict[str, Any] | None, n: int = 8) -> list[str]:
    if not credits or "cast" not in credits:
        return []
    return [c.get("name", "") for c in credits.get("cast", [])[:n] if c.get("name")]


def extract_certifications(release_dates_payload: dict[str, Any] | None) -> list[str]:
    """Pull certification strings (e.g. PG-13, R) from release_dates object."""
    if not release_dates_payload or "results" not in release_dates_payload:
        return []
    certs: set[str] = set()
    for country in release_dates_payload.get("results", []):
        for rd in country.get("release_dates", []):
            c = rd.get("certification")
            if c:
                certs.add(str(c).strip())
    return sorted(certs)


def us_certification_from_details(movie: dict[str, Any] | None) -> str | None:
    """Best-effort US theatrical certification from appended release_dates."""
    rd = None
    if movie and "release_dates" in movie:
        rd = movie["release_dates"]
    if not rd or "results" not in rd:
        return None
    for country in rd.get("results", []):
        if country.get("iso_3166_1") != "US":
            continue
        for item in country.get("release_dates", []):
            cert = item.get("certification")
            typ = item.get("type")
            if cert and typ in (2, 3):  # theatrical / digital typical types
                return cert
        for item in country.get("release_dates", []):
            cert = item.get("certification")
            if cert:
                return cert
    return None
