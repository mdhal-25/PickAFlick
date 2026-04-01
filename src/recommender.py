"""
Content-based TF-IDF recommendations with hybrid sentiment + popularity weighting.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import api
from src import filters as filter_mod
from src import nlp_utils

# Two-stage ranking weights (content / sentiment / genreOverlap / popularity)
W_CONTENT = 0.65
W_SENTIMENT = 0.20
W_GENRE = 0.10
W_POPULARITY = 0.05

DEFAULT_CONTENT_SIM_THRESHOLD = 0.15
FALLBACK_CONTENT_SIM_THRESHOLD = 0.12

WEAK_GENERIC_WORDS: set[str] = {"film", "movie", "story", "life", "world"}
SUPERHERO_GENRE_IDS: set[int] = {28, 12, 14, 878}  # Action / Adventure / Fantasy / Sci-Fi
SUPERHERO_REQUIRED_SIGNALS: tuple[str, ...] = (
    "superhero",
    "super hero",
    "vigilante",
    "masked hero",
    "crime fighter",
    "save the world",
    "supervillain",
    "comic book",
    "marvel",
    "dc comics",
    "avengers",
    "batman",
    "spider-man",
    "spiderman",
    "x-men",
)
SUPERHERO_WEAK_SIGNALS: tuple[str, ...] = (
    "powers",
    "power",
    "abilities",
    "ability",
    "mutation",
    "mutant",
    "enhanced",
    "superpower",
    "super power",
)

ANIMAL_THEME_TERMS: tuple[str, ...] = (
    "animal",
    "animals",
    "dog",
    "cat",
    "lion",
    "tiger",
    "wolf",
    "bear",
    "pet",
    "wildlife",
    "creature",
    "talking animal",
    "zoo",
    "jungle",
    "farm",
    "horse",
    "bird",
    "shark",
    "fish",
)
ANIMAL_GENRE_IDS: set[int] = {10751, 16, 12, 14, 35}  # Family/Animation/Adventure/Fantasy/Comedy
MUSIC_THEME_TERMS: tuple[str, ...] = (
    "singing",
    "songs",
    "song",
    "musical",
    "music",
    "performer",
    "concert",
    "stage",
    "singer",
    "band",
    "talent show",
    "audition",
    "competition",
    "performance",
    "stage performance",
)
MUSIC_GENRE_IDS: set[int] = {10402, 16, 10751, 35, 14}  # Music + common family musical adjacencies


def _clean_text(s: str) -> str:
    t = (s or "").lower()
    for w in WEAK_GENERIC_WORDS:
        t = re.sub(rf"\b{re.escape(w)}\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _repeat(xs: list[str], times: int) -> list[str]:
    if times <= 1:
        return xs
    return xs * times


def extract_genre_names(movie: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for g in movie.get("genres") or []:
        if isinstance(g, dict) and g.get("name"):
            out.append(str(g["name"]))
    return out


def extract_keyword_names(movie: dict[str, Any]) -> list[str]:
    out: list[str] = []
    kw_block = movie.get("keywords")
    if isinstance(kw_block, dict):
        for k in kw_block.get("keywords") or []:
            if isinstance(k, dict) and k.get("name"):
                out.append(str(k["name"]))
    return out


def build_combined_text(movie: dict[str, Any]) -> str:
    """
    Flatten genres, keywords, director, top cast, and overview into one string
    for TF-IDF. Missing sections are skipped safely.

    Overview text is intentionally down-weighted relative to structured fields.
    """
    genres = extract_genre_names(movie)
    keywords = extract_keyword_names(movie)

    directors: list[str] = []
    cast: list[str] = []
    studios: list[str] = []
    credits = movie.get("credits")
    if isinstance(credits, dict):
        directors = api.extract_directors(credits)
        cast = api.extract_top_cast(credits, n=10)
    studios = [
        str(c.get("name"))
        for c in (movie.get("production_companies") or [])
        if isinstance(c, dict) and c.get("name")
    ]

    overview = _clean_text(str(movie.get("overview") or ""))

    # Weight structured fields higher by repetition.
    chunks: list[str] = []
    chunks.extend(_repeat([_clean_text(x) for x in genres if x], 3))
    chunks.extend(_repeat([_clean_text(x) for x in keywords if x], 3))
    chunks.extend(_repeat([_clean_text(x) for x in directors if x], 2))
    chunks.extend(_repeat([_clean_text(x) for x in cast if x], 2))
    chunks.extend(_repeat([_clean_text(x) for x in studios if x], 2))
    if overview:
        chunks.append(overview)

    return " ".join([c for c in chunks if c]).strip()


def _popularity_component(movie: dict[str, Any], max_votes: float) -> float:
    """Map vote average + vote count to [0, 1]."""
    va = movie.get("vote_average")
    vc = movie.get("vote_count") or 0
    try:
        va_f = float(va) if va is not None else 0.0
    except (TypeError, ValueError):
        va_f = 0.0
    va_f = max(0.0, min(10.0, va_f)) / 10.0

    vc_i = max(0, int(vc))
    denom = math.log1p(max(max_votes, 1.0))
    vc_norm = math.log1p(vc_i) / denom if denom > 0 else 0.0
    vc_norm = float(max(0.0, min(1.0, vc_norm)))

    return 0.55 * va_f + 0.45 * vc_norm


def _content_similarities(anchor_text: str, candidate_texts: list[str]) -> np.ndarray:
    """Cosine similarity between anchor and each candidate using one shared TF-IDF fit."""
    corpus = [anchor_text] + candidate_texts
    if not any(corpus):
        return np.zeros(len(candidate_texts))

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
    )
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # Degenerate corpus (e.g. all empty after tokenization)
        return np.zeros(len(candidate_texts))

    sims = cosine_similarity(matrix[0:1], matrix[1:])
    return np.asarray(sims).flatten()


def _genre_ids(movie: dict[str, Any]) -> set[int]:
    out: set[int] = set()
    for g in movie.get("genres") or []:
        if isinstance(g, dict) and g.get("id"):
            out.add(int(g["id"]))
    return out


def genre_overlap_count(a: dict[str, Any], b: dict[str, Any]) -> int:
    return len(_genre_ids(a) & _genre_ids(b))


def compute_genre_overlap_score(overlap_count: int) -> float:
    """
    Explicit genre overlap score in [0, 1].
    - 0 overlap -> excluded upstream (or score 0)
    - 1 overlap -> medium
    - 2+ overlap -> high
    """
    if overlap_count <= 0:
        return 0.0
    if overlap_count == 1:
        return 0.6
    return 1.0


def _shared_genre_names(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    a_names = {str(g.get("name")) for g in (a.get("genres") or []) if isinstance(g, dict) and g.get("name")}
    b_names = {str(g.get("name")) for g in (b.get("genres") or []) if isinstance(g, dict) and g.get("name")}
    return sorted(list(a_names & b_names))


def _shared_keywords(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    a_kw = {k.lower() for k in extract_keyword_names(a) if k}
    b_kw = {k.lower() for k in extract_keyword_names(b) if k}
    shared = sorted(list(a_kw & b_kw))
    return shared[:8]


def _normalize_title(title: str) -> str:
    t = (title or "").lower().strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t


def is_superhero_movie(movie: dict[str, Any]) -> bool:
    """
    Strict superhero check.
    Returns True ONLY if at least one REQUIRED superhero/comic/vigilante/franchise signal exists
    in (title + overview + keywords + collection name).
    """
    title_l = str(movie.get("title") or "").lower()
    overview_l = str(movie.get("overview") or "").lower()
    kws_l = " ".join([k.lower() for k in extract_keyword_names(movie) if k]).lower()
    col_l = ""
    bc = movie.get("belongs_to_collection")
    if isinstance(bc, dict):
        col_l = str(bc.get("name") or "").lower()

    text = " ".join([title_l, overview_l, kws_l, col_l]).strip()
    return any(sig in text for sig in SUPERHERO_REQUIRED_SIGNALS)


def _has_weak_superhero_signal(movie: dict[str, Any]) -> bool:
    """
    Weak superhero signal: requires at least one weak keyword AND superhero-adjacent genres.
    This prevents dramas with 'abilities' from being treated as superhero.
    """
    text = " ".join(
        [
            str(movie.get("title") or "").lower(),
            str(movie.get("overview") or "").lower(),
            " ".join([k.lower() for k in extract_keyword_names(movie) if k]).lower(),
        ]
    )
    if not any(sig in text for sig in SUPERHERO_WEAK_SIGNALS):
        return False
    gids = _genre_ids(movie)
    return len(gids & SUPERHERO_GENRE_IDS) > 0


def score_animal_theme(movie: dict[str, Any]) -> int:
    """
    Broad animal theme score with centrality weighting.
    Higher when animals are central in title/overview/keywords.
    """
    title_l = str(movie.get("title") or "").lower()
    overview_l = str(movie.get("overview") or "").lower()
    kws = [k.lower() for k in extract_keyword_names(movie) if k]
    kw_l = " ".join(kws)

    title_hits = sum(1 for t in ANIMAL_THEME_TERMS if t in title_l)
    kw_hits = sum(1 for t in ANIMAL_THEME_TERMS if t in kw_l)
    ov_hits = sum(1 for t in ANIMAL_THEME_TERMS if t in overview_l)

    score = 0
    score += min(3, title_hits * 2)           # title is strongest centrality clue
    score += min(3, kw_hits * 2)              # explicit keywords are strong
    score += min(2, ov_hits)                  # overview is supportive
    if len(_genre_ids(movie) & ANIMAL_GENRE_IDS) >= 2:
        score += 1
    return int(max(0, min(7, score)))


def apply_animal_tiered_filter(
    movies: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int | str]]:
    """
    Animal theme fallback tiers:
    strict: score>=2
    partial: score>=1
    soft: animal-adjacent genres (Family/Animation/Adventure/Fantasy/Comedy)
    """
    if not movies:
        return [], {"retrieval": 0, "strict": 0, "partial": 0, "soft": 0, "tier_used": "none"}
    scored = [(m, score_animal_theme(m)) for m in movies]
    strict = [m for (m, s) in scored if s >= 2]
    partial = [m for (m, s) in scored if s >= 1]
    soft = [m for (m, _s) in scored if len(_genre_ids(m) & ANIMAL_GENRE_IDS) > 0]
    if strict:
        chosen = strict
        tier = "strict"
    elif partial:
        chosen = partial
        tier = "partial"
    elif soft:
        chosen = soft
        tier = "soft"
    else:
        chosen = [m for (m, _s) in scored]
        tier = "fallback"
    max_votes = float(max((m.get("vote_count") or 0) for m in chosen) or 1.0) if chosen else 1.0
    chosen.sort(key=lambda m: (score_animal_theme(m), _popularity_component(m, max_votes)), reverse=True)
    return chosen, {
        "retrieval": len(movies),
        "strict": len(strict),
        "partial": len(partial),
        "soft": len(soft),
        "tier_used": tier,
    }


def score_music_theme(movie: dict[str, Any]) -> int:
    """
    Music/singing/performance theme score with centrality weighting.
    """
    title_l = str(movie.get("title") or "").lower()
    overview_l = str(movie.get("overview") or "").lower()
    kws = [k.lower() for k in extract_keyword_names(movie) if k]
    kw_l = " ".join(kws)

    title_hits = sum(1 for t in MUSIC_THEME_TERMS if t in title_l)
    kw_hits = sum(1 for t in MUSIC_THEME_TERMS if t in kw_l)
    ov_hits = sum(1 for t in MUSIC_THEME_TERMS if t in overview_l)

    score = 0
    score += min(3, title_hits * 2)
    score += min(3, kw_hits * 2)
    score += min(2, ov_hits)
    if 10402 in _genre_ids(movie):
        score += 2
    elif len(_genre_ids(movie) & MUSIC_GENRE_IDS) >= 2:
        score += 1
    return int(max(0, min(8, score)))


def popularity_confidence_score(movie: dict[str, Any], max_votes_hint: float | None = None) -> float:
    """
    Public popularity confidence in [0,1] blending TMDB vote_count + popularity + rating.
    """
    votes = float(movie.get("vote_count") or 0.0)
    pop = float(movie.get("popularity") or 0.0)
    va = float(movie.get("vote_average") or 0.0) / 10.0
    max_votes = float(max_votes_hint or max(1.0, votes))
    votes_norm = math.log1p(votes) / max(1.0, math.log1p(max_votes))
    pop_norm = min(1.0, pop / 120.0)  # pragmatic TMDB popularity normalization
    return float(max(0.0, min(1.0, 0.5 * votes_norm + 0.3 * pop_norm + 0.2 * va)))


def _theme_signal_scores(movie: dict[str, Any], terms: list[str]) -> tuple[float, float]:
    """
    Returns (theme_match_score, centrality_score) in [0,1].
    Centrality favors title/keywords over incidental overview mentions.
    """
    if not terms:
        return 0.0, 0.0
    title_l = str(movie.get("title") or "").lower()
    overview_l = str(movie.get("overview") or "").lower()
    kws_l = " ".join([k.lower() for k in extract_keyword_names(movie) if k]).lower()
    col = movie.get("belongs_to_collection")
    col_l = str(col.get("name") or "").lower() if isinstance(col, dict) else ""

    tset = [t.lower() for t in terms if t]
    title_hits = sum(1 for t in tset if t in title_l)
    kw_hits = sum(1 for t in tset if t in kws_l)
    ov_hits = sum(1 for t in tset if t in overview_l)
    col_hits = sum(1 for t in tset if t in col_l)

    coverage = min(1.0, (title_hits + kw_hits + ov_hits + col_hits) / max(1.0, len(tset)))
    central_raw = (2.4 * title_hits) + (2.0 * kw_hits) + (1.4 * col_hits) + (0.8 * ov_hits)
    centrality = min(1.0, central_raw / max(1.0, 2.0 * len(tset)))
    return float(coverage), float(centrality)


def _genre_match_score(movie: dict[str, Any], explicit_genre_ids: list[int]) -> float:
    if not explicit_genre_ids:
        return 0.5
    mids = _genre_ids(movie)
    overlap = len(mids & set(int(g) for g in explicit_genre_ids))
    return min(1.0, overlap / max(1.0, len(explicit_genre_ids)))


def score_candidate(movie: dict[str, Any], intent: nlp_utils.QueryIntent, *, max_votes_hint: float) -> tuple[float, dict[str, float]]:
    """
    General multi-factor scorer:
    final = 0.25 genre + 0.25 theme + 0.20 centrality + 0.10 franchise + 0.10 tone + 0.10 popularity - penalties
    """
    genre_score = _genre_match_score(movie, intent.explicit_genre_ids)

    # Aggregate themes across buckets.
    theme_scores: list[float] = []
    central_scores: list[float] = []
    for _bucket, terms in (intent.themes or {}).items():
        ts, cs = _theme_signal_scores(movie, terms)
        theme_scores.append(ts)
        central_scores.append(cs)
    theme_match = float(sum(theme_scores) / len(theme_scores)) if theme_scores else 0.5
    centrality = float(sum(central_scores) / len(central_scores)) if central_scores else 0.5

    # Exclusions: keep hard behavior for explicit exclusions.
    exclusion_penalty = float(
        filter_mod.exclusion_penalty_for_intent(
            movie,
            excluded_genre_ids=intent.excluded_genre_ids,
            exclude_superhero=bool(intent.exclude_superhero),
        )
    )

    if intent.required_franchise:
        franchise_match = 1.0 if filter_mod.matches_franchise(movie, intent.required_franchise) else 0.0
    else:
        franchise_match = 0.5

    # Tone by sentiment polarity compatibility.
    sent = float(nlp_utils.overview_sentiment(movie.get("overview")))
    if intent.mood == "positive":
        tone_match = float(nlp_utils.sentiment_similarity(0.55, sent))
    elif intent.mood == "negative":
        tone_match = float(nlp_utils.sentiment_similarity(-0.55, sent))
    else:
        tone_match = 0.5

    pop_conf = popularity_confidence_score(movie, max_votes_hint=max_votes_hint)
    final = (
        (0.25 * genre_score)
        + (0.25 * theme_match)
        + (0.20 * centrality)
        + (0.10 * franchise_match)
        + (0.10 * tone_match)
        + (0.10 * pop_conf)
        - (0.25 * exclusion_penalty)
    )
    final = float(max(0.0, min(1.5, final)))
    comps = {
        "genre_match_score": float(genre_score),
        "theme_match_score": float(theme_match),
        "centrality_score": float(centrality),
        "franchise_match_score": float(franchise_match),
        "tone_match_score": float(tone_match),
        "popularity_confidence_score": float(pop_conf),
        "exclusion_penalty": float(exclusion_penalty),
        "hybrid": float(final),
    }
    return final, comps


def rank_candidates(
    candidates: list[dict[str, Any]],
    intent: nlp_utils.QueryIntent,
    *,
    top_n: int = 30,
    anchor: dict[str, Any] | None = None,
    relationship_context: dict[str, Any] | None = None,
) -> list[tuple[dict[str, Any], float, dict[str, float]]]:
    """General ranking model over broad candidate set."""
    if not candidates:
        return []
    max_votes = float(max((m.get("vote_count") or 0) for m in candidates) or 1.0)
    scored: list[tuple[dict[str, Any], float, dict[str, float]]] = []
    for m in candidates:
        s, comps = score_candidate(m, intent, max_votes_hint=max_votes)
        rel_s, rel_notes = relationship_score(m, anchor, relationship_context)
        if rel_s > 0:
            s = float(s + (0.16 * rel_s))
            comps["relationship_score"] = float(rel_s)
            comps["tmdb_same_collection"] = 1.0 if "same_collection" in rel_notes else 0.0
            comps["tmdb_similar"] = 1.0 if "tmdb_similar" in rel_notes else 0.0
            comps["tmdb_recommendation"] = 1.0 if "tmdb_recommendation" in rel_notes else 0.0
        # Keep explicit exclusion handling hard.
        if comps.get("exclusion_penalty", 0.0) >= 1.0:
            continue
        # Keep franchise/entity constraints hard when present.
        if intent.required_franchise and comps.get("franchise_match_score", 0.0) < 1.0:
            continue
        scored.append((m, s, comps))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(1, int(top_n))]


def split_ranked_tiers(
    ranked: list[tuple[dict[str, Any], float, dict[str, float]]],
) -> tuple[
    list[tuple[dict[str, Any], float, dict[str, float]]],
    list[tuple[dict[str, Any], float, dict[str, float]]],
    list[tuple[dict[str, Any], float, dict[str, float]]],
]:
    """Tiered fallback split: strong / close / broad."""
    exact: list[tuple[dict[str, Any], float, dict[str, float]]] = []
    close: list[tuple[dict[str, Any], float, dict[str, float]]] = []
    broad: list[tuple[dict[str, Any], float, dict[str, float]]] = []
    for m, s, c in ranked:
        if s >= 0.72:
            exact.append((m, s, {**c, "match_level": "exact"}))
        elif s >= 0.56:
            close.append((m, s, {**c, "match_level": "closest"}))
        else:
            broad.append((m, s, {**c, "match_level": "closest"}))
    return exact, close, broad


def canonical_theme_boost(
    movie: dict[str, Any],
    *,
    query_text: str,
    require_animal_theme: bool,
    require_music_theme: bool,
) -> float:
    """
    Boost obvious cultural prototypes (e.g. Sing/Sing 2 for singing-animals intent).
    """
    if not (require_animal_theme and require_music_theme):
        return 0.0
    q = (query_text or "").lower()
    if not any(t in q for t in ("sing", "singing", "musical", "songs", "music")):
        return 0.0
    title_l = str(movie.get("title") or "").lower()
    col = movie.get("belongs_to_collection")
    col_l = str(col.get("name") or "").lower() if isinstance(col, dict) else ""
    animal_sc = score_animal_theme(movie)
    music_sc = score_music_theme(movie)
    pop_sc = popularity_confidence_score(movie)

    # Direct archetype titles (kept narrow on purpose)
    if re.search(r"\bsing(?:\s*2)?\b", title_l) or re.search(r"\bsing\b", col_l):
        return 1.0
    # Generic archetype: both themes strong + socially validated title
    if animal_sc >= 4 and music_sc >= 4 and pop_sc >= 0.55:
        return 0.45
    return 0.0


def apply_broad_theme_tiered_filter(
    movies: list[dict[str, Any]],
    *,
    require_animal_theme: bool,
    require_music_theme: bool,
) -> tuple[list[dict[str, Any]], dict[str, int | str]]:
    """
    Combined broad-theme gate/ranker for queries like "singing animals".
    Prioritizes candidates matching both themes; falls back progressively.
    """
    if not movies:
        return [], {"retrieval": 0, "strict": 0, "partial": 0, "soft": 0, "tier_used": "none"}
    if not require_animal_theme and not require_music_theme:
        return movies, {"retrieval": len(movies), "strict": len(movies), "partial": len(movies), "soft": len(movies), "tier_used": "none"}

    animal_scores = [score_animal_theme(m) for m in movies]
    music_scores = [score_music_theme(m) for m in movies]
    strict: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    soft: list[dict[str, Any]] = []

    for i, m in enumerate(movies):
        a = animal_scores[i]
        mu = music_scores[i]
        has_a = a >= 2
        has_m = mu >= 2
        strong_a = a >= 4
        strong_m = mu >= 4
        if require_animal_theme and require_music_theme:
            if strong_a and strong_m:
                strict.append(m)
            if has_a and has_m:
                partial.append(m)
            if has_a or has_m:
                soft.append(m)
        elif require_animal_theme:
            if strong_a:
                strict.append(m)
            if has_a:
                partial.append(m)
            if has_a or (len(_genre_ids(m) & ANIMAL_GENRE_IDS) > 0):
                soft.append(m)
        elif require_music_theme:
            if strong_m:
                strict.append(m)
            if has_m:
                partial.append(m)
            if has_m or (len(_genre_ids(m) & MUSIC_GENRE_IDS) > 0):
                soft.append(m)

    if strict:
        chosen = strict
        tier = "strict"
    elif partial:
        chosen = partial
        tier = "partial"
    elif soft:
        chosen = soft
        tier = "soft"
    else:
        chosen = list(movies)
        tier = "fallback"

    max_votes = float(max((m.get("vote_count") or 0) for m in chosen) or 1.0)
    # Combined ranking: both themes first, then confidence
    chosen.sort(
        key=lambda m: (
            score_animal_theme(m) + score_music_theme(m),
            min(score_animal_theme(m), score_music_theme(m)) if (require_animal_theme and require_music_theme) else 0,
            1 if (score_animal_theme(m) >= 4 and score_music_theme(m) >= 4) else 0,
            _popularity_component(m, max_votes),
        ),
        reverse=True,
    )
    return chosen, {
        "retrieval": len(movies),
        "strict": len(strict),
        "partial": len(partial),
        "soft": len(soft),
        "tier_used": tier,
    }


def apply_superhero_tiered_filter(
    movies: list[dict[str, Any]],
    *,
    require_romance: bool,
) -> tuple[list[dict[str, Any]], dict[str, int | str]]:
    """
    Tiered superhero matching:
    1) strict: superhero_score>=2 (+ romance when required)
    2) partial: superhero_score>=1 (+ romance when required)
    3) soft: superhero-adjacent genres (+ romance when required)
    """
    if not movies:
        return [], {"retrieval": 0, "strict": 0, "partial": 0, "soft": 0, "tier_used": "none"}

    def romance_ok(m: dict[str, Any]) -> bool:
        if not require_romance:
            return True
        return 10749 in _genre_ids(m)  # Romance

    strict = [m for m in movies if is_superhero_movie(m) and romance_ok(m)]
    weak = [m for m in movies if _has_weak_superhero_signal(m) and romance_ok(m)]
    soft = [m for m in movies if (len(_genre_ids(m) & SUPERHERO_GENRE_IDS) > 0) and romance_ok(m)]

    if strict:
        chosen = strict
        tier = "strict"
    elif weak:
        # fallback: allow action/sci-fi/fantasy/adventure, but require at least one weak superhero signal
        chosen = weak
        tier = "weak"
    elif soft:
        chosen = soft
        tier = "soft"
    else:
        chosen = [m for m in movies if romance_ok(m)] if require_romance else list(movies)
        tier = "fallback"

    max_votes = float(max((m.get("vote_count") or 0) for m in chosen) or 1.0) if chosen else 1.0
    chosen.sort(key=lambda m: (_popularity_component(m, max_votes)), reverse=True)
    return chosen, {
        "retrieval": len(movies),
        "strict": len(strict),
        "partial": len(weak),
        "soft": len(soft),
        "tier_used": tier,
    }


def detect_franchise_relationship(anchor: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, str | None]:
    """
    Returns (boost_score, explanation_text_if_high_confidence).
    High confidence from same TMDB collection; otherwise conservative title heuristic.
    """
    a_col = anchor.get("belongs_to_collection") if isinstance(anchor, dict) else None
    c_col = candidate.get("belongs_to_collection") if isinstance(candidate, dict) else None
    if isinstance(a_col, dict) and isinstance(c_col, dict):
        if a_col.get("id") and c_col.get("id") and a_col.get("id") == c_col.get("id"):
            return 1.0, "This title belongs to the same TMDB collection as the anchor movie."

    at = _normalize_title(str(anchor.get("title") or ""))
    ct = _normalize_title(str(candidate.get("title") or ""))
    if at and ct:
        # conservative continuation heuristic (shared stem + numbering/subtitle)
        a_tokens = at.split()
        c_tokens = ct.split()
        shared_prefix = 0
        for x, y in zip(a_tokens, c_tokens):
            if x == y:
                shared_prefix += 1
            else:
                break
        has_seq_hint = bool(re.search(r"\b(2|3|4|ii|iii|iv|part)\b", ct))
        if shared_prefix >= 2 and has_seq_hint:
            return 0.7, "This appears to be a sequel or continuation of the same series."
        if shared_prefix >= 3:
            return 0.45, "This appears closely related to the same franchise naming pattern."

    # Additional franchise hint: strong shared keyword/theme signatures.
    shared_kw = _shared_keywords(anchor, candidate)
    if len(shared_kw) >= 3:
        return 0.6, None
    if len(shared_kw) >= 2:
        # If titles also share a strong prefix, raise confidence.
        if at and ct and len(set(at.split()[:3]) & set(ct.split()[:3])) >= 2:
            return 0.7, None
    return 0.0, None


def relationship_score(
    movie: dict[str, Any],
    anchor: dict[str, Any] | None,
    relationship_context: dict[str, Any] | None,
) -> tuple[float, list[str]]:
    """
    Generic TMDB relationship score for anchor-aware ranking:
    - same collection
    - membership in TMDB similar list
    - membership in TMDB recommendations list
    - sequel/follow-up heuristic
    """
    if not anchor:
        return 0.0, []
    mid = int(movie.get("id") or 0)
    if mid <= 0:
        return 0.0, []

    rel = relationship_context or {}
    collection_member_ids = {int(x) for x in (rel.get("collection_member_ids") or set()) if isinstance(x, int)}
    similar_ids = {int(x) for x in (rel.get("similar_ids") or set()) if isinstance(x, int)}
    recommendation_ids = {int(x) for x in (rel.get("recommendation_ids") or set()) if isinstance(x, int)}

    notes: list[str] = []
    score = 0.0
    if mid in collection_member_ids:
        score += 1.35
        notes.append("same_collection")
    if mid in similar_ids:
        score += 0.70
        notes.append("tmdb_similar")
    if mid in recommendation_ids:
        score += 0.55
        notes.append("tmdb_recommendation")

    seq_score, seq_note = detect_franchise_relationship(anchor, movie)
    if seq_score > 0:
        score += 0.35 * float(seq_score)
        notes.append("sequel_followup")
        if seq_note:
            notes.append(seq_note)
    return float(min(2.0, score)), notes


def _query_metadata_boost(query_terms: list[str], movie: dict[str, Any]) -> float:
    """
    Additional relevance for studio/keyword/title/genre term matches.
    Helps queries like 'Pixar animation' or franchise/theme terms.
    """
    if not query_terms:
        return 0.0
    title = _clean_text(str(movie.get("title") or ""))
    genres = [_clean_text(g) for g in extract_genre_names(movie)]
    keywords = [_clean_text(k) for k in extract_keyword_names(movie)]
    studios = [
        _clean_text(str(c.get("name")))
        for c in (movie.get("production_companies") or [])
        if isinstance(c, dict) and c.get("name")
    ]
    hay = " ".join([title] + genres + keywords + studios)
    if not hay:
        return 0.0
    hits = 0
    for t in query_terms:
        tt = _clean_text(t)
        if tt and tt in hay:
            hits += 1
    return min(1.0, hits / max(1.0, len(query_terms)))


def hybrid_recommendations(
    anchor: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 15,
    mood_sentiment_target: float | None = None,
    content_sim_threshold: float = DEFAULT_CONTENT_SIM_THRESHOLD,
    w_content: float = W_CONTENT,
    w_sentiment: float = W_SENTIMENT,
    w_genre: float = W_GENRE,
    w_popularity: float = W_POPULARITY,
    relationship_context: dict[str, Any] | None = None,
) -> list[tuple[dict[str, Any], float, dict[str, float]]]:
    """
    Rank candidates by weighted mix of TF-IDF cosine, sentiment alignment, and popularity.

    Returns list of (movie, score, components) sorted descending by score.
    components keys: content, sentiment, popularity, hybrid

    If mood_sentiment_target is set, sentiment term blends anchor overview alignment
    with the target mood polarity so NL queries like "uplifting" nudge results.
    """
    if not candidates:
        return []

    anchor_id = anchor.get("id")
    # HARD GENRE FILTER (exclude 0-overlap candidates) + remove self
    filtered: list[dict[str, Any]] = []
    overlaps: list[int] = []
    for m in candidates:
        if m.get("id") == anchor_id:
            continue
        ov = genre_overlap_count(anchor, m)
        if ov <= 0:
            continue
        filtered.append(m)
        overlaps.append(ov)
    if not filtered:
        return []

    anchor_text = build_combined_text(anchor)
    cand_texts = [build_combined_text(m) for m in filtered]

    content_raw = _content_similarities(anchor_text, cand_texts)

    anchor_sent = nlp_utils.overview_sentiment(anchor.get("overview"))
    cand_sent = np.array(
        [nlp_utils.overview_sentiment(m.get("overview")) for m in filtered],
        dtype=float,
    )

    sent_align = np.array(
        [
            nlp_utils.sentiment_similarity(anchor_sent, cs)
            for cs in cand_sent
        ],
        dtype=float,
    )

    if mood_sentiment_target is not None:
        mood_align = np.array(
            [
                nlp_utils.sentiment_similarity(mood_sentiment_target, cs)
                for cs in cand_sent
            ],
            dtype=float,
        )
        sent_align = 0.65 * sent_align + 0.35 * mood_align

    max_votes = float(max((m.get("vote_count") or 0) for m in filtered) or 1.0)
    pops = np.array([_popularity_component(m, max_votes) for m in filtered], dtype=float)

    # CONTENT THRESHOLD (stage-1); can be relaxed in fallback
    keep_idx = np.where(content_raw > float(content_sim_threshold))[0]
    if keep_idx.size == 0:
        return []

    content_kept = content_raw[keep_idx]
    sent_kept = sent_align[keep_idx]
    pop_kept = pops[keep_idx]
    overlaps_kept = np.array([overlaps[int(i)] for i in keep_idx], dtype=float)
    genre_scores = np.array([compute_genre_overlap_score(int(x)) for x in overlaps_kept], dtype=float)

    hybrid = (w_content * content_kept) + (w_sentiment * sent_kept) + (w_genre * genre_scores) + (w_popularity * pop_kept)
    hybrid = np.clip(hybrid, 0.0, 1.2)

    order = np.argsort(hybrid)[::-1][: max(1, int(top_n))]
    out: list[tuple[dict[str, Any], float, dict[str, float]]] = []
    for idx in order:
        j = int(idx)
        i = int(keep_idx[j])
        movie = filtered[i]
        shared_genres = _shared_genre_names(anchor, movie)
        shared_kw = _shared_keywords(anchor, movie)
        franchise_score, franchise_note = detect_franchise_relationship(anchor, movie)
        rel_s, rel_notes = relationship_score(movie, anchor, relationship_context)
        hybrid_boosted = float(hybrid[j]) + (0.18 * franchise_score) + (0.34 * rel_s)
        comps = {
            "content": float(content_raw[i]),
            "sentiment": float(sent_align[i]),
            "popularity": float(pops[i]),
            "genre_overlap": float(overlaps[i]),
            "genre_overlap_score": float(compute_genre_overlap_score(int(overlaps[i]))),
            "franchise_score": float(franchise_score),
            "franchise_note": franchise_note or "",
            "relationship_score": float(rel_s),
            "tmdb_same_collection": 1.0 if "same_collection" in rel_notes else 0.0,
            "tmdb_similar": 1.0 if "tmdb_similar" in rel_notes else 0.0,
            "tmdb_recommendation": 1.0 if "tmdb_recommendation" in rel_notes else 0.0,
            "hybrid": float(max(0.0, min(1.2, hybrid_boosted))),
        }
        # attach lightweight fields used for explanations (kept in comps)
        comps["shared_genres_count"] = float(len(shared_genres))
        comps["shared_keywords_count"] = float(len(shared_kw))
        out.append((movie, float(max(0.0, min(1.2, hybrid_boosted))), comps))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def query_recommendations(
    query_text: str,
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 18,
    mood_sentiment_target: float | None = None,
    requested_genre_ids: set[int] | None = None,
    content_sim_threshold: float = DEFAULT_CONTENT_SIM_THRESHOLD,
) -> list[tuple[dict[str, Any], float, dict[str, float]]]:
    """
    Anchor-free recommendations: compute TF-IDF cosine similarity between a query document
    and candidate combined texts, then apply the same thresholding and hybrid blend.
    """
    if not candidates:
        return []
    qdoc = _clean_text(query_text or "")
    if not qdoc:
        return []
    query_terms = [t for t in re.split(r"\s+", qdoc) if t]

    cand_texts = [build_combined_text(m) for m in candidates]
    content_raw = _content_similarities(qdoc, cand_texts)
    keep_idx = np.where(content_raw > float(content_sim_threshold))[0]
    if keep_idx.size == 0:
        return []

    sent_scores = np.array([nlp_utils.overview_sentiment(m.get("overview")) for m in candidates], dtype=float)
    if mood_sentiment_target is not None:
        sent_align = np.array(
            [nlp_utils.sentiment_similarity(mood_sentiment_target, float(s)) for s in sent_scores],
            dtype=float,
        )
    else:
        # if no mood, don't artificially boost/penalize: neutral 0.5 everywhere
        sent_align = np.ones_like(sent_scores) * 0.5

    max_votes = float(max((m.get("vote_count") or 0) for m in candidates) or 1.0)
    pops = np.array([_popularity_component(m, max_votes) for m in candidates], dtype=float)

    # Explicit genre overlap score against requested genres (0 overlap -> exclude)
    if requested_genre_ids:
        overlap_counts = np.array(
            [len(_genre_ids(m) & requested_genre_ids) for m in candidates],
            dtype=float,
        )
        keep_genre = np.where(overlap_counts > 0)[0]
        keep_idx = np.intersect1d(keep_idx, keep_genre, assume_unique=False)
        if keep_idx.size == 0:
            return []
        overlap_kept = overlap_counts[keep_idx]
        genre_scores = np.array([compute_genre_overlap_score(int(x)) for x in overlap_kept], dtype=float)
    else:
        genre_scores = np.ones(len(keep_idx), dtype=float) * 0.5

    content_kept = content_raw[keep_idx]
    sent_kept = sent_align[keep_idx]
    pop_kept = pops[keep_idx]
    meta_boost_all = np.array([_query_metadata_boost(query_terms, m) for m in candidates], dtype=float)
    meta_kept = meta_boost_all[keep_idx]

    # Query mode tuning: slightly favor metadata hits for studio/franchise terms.
    hybrid = (
        0.58 * content_kept
        + 0.17 * sent_kept
        + 0.15 * genre_scores
        + 0.05 * pop_kept
        + 0.05 * meta_kept
    )
    hybrid = np.clip(hybrid, 0.0, 1.2)

    order = np.argsort(hybrid)[::-1][: max(1, int(top_n))]
    out: list[tuple[dict[str, Any], float, dict[str, float]]] = []
    for idx in order:
        j = int(idx)
        i = int(keep_idx[j])
        m = candidates[i]
        comps = {
            "content": float(content_raw[i]),
            "sentiment": float(sent_align[i]),
            "popularity": float(pops[i]),
            "genre_overlap_score": float(genre_scores[j]) if genre_scores.size == keep_idx.size else 0.5,
            "query_metadata_boost": float(meta_kept[j]) if meta_kept.size == keep_idx.size else 0.0,
            "hybrid": float(hybrid[j]),
        }
        out.append((m, float(hybrid[j]), comps))
    return out


def explanation_blurb(
    anchor: dict[str, Any] | None,
    candidate: dict[str, Any],
    components: dict[str, float] | None,
) -> str:
    """Cleaner explanation: shared genres, shared keywords/themes, and similar tone."""
    title = candidate.get("title") or "This title"
    if anchor:
        shared_genres = _shared_genre_names(anchor, candidate)
        shared_kw = _shared_keywords(anchor, candidate)
        tone = nlp_utils.sentiment_similarity(
            nlp_utils.overview_sentiment(anchor.get("overview")),
            nlp_utils.overview_sentiment(candidate.get("overview")),
        )
    else:
        # Query-only / genre-only: explain using the candidate's strongest signals.
        shared_genres = extract_genre_names(candidate)[:3]
        shared_kw = extract_keyword_names(candidate)[:4]
        tone = nlp_utils.sentiment_similarity(0.0, nlp_utils.overview_sentiment(candidate.get("overview")))

    match_level = (components or {}).get("match_level")
    if match_level == "closest_safe":
        prefix = "is a closest match that prioritizes family-safe constraints"
    elif match_level == "closest":
        prefix = "is a closest match based on your constraints"
    else:
        prefix = "matches because"

    parts: list[str] = []
    if components and components.get("required_franchise"):
        parts.append(f"it matches your **{components.get('required_franchise')}** franchise constraint")
    if components and components.get("required_superhero_theme", 0.0) >= 1.0:
        parts.append("it matches your **superhero** theme requirement")
    if components and components.get("required_animal_theme", 0.0) >= 1.0:
        parts.append("it matches your **animal/creature** theme request")
    if components and components.get("required_music_theme", 0.0) >= 1.0:
        parts.append("it matches your **music/singing** theme request")
    if components and components.get("tmdb_same_collection", 0.0) >= 1.0:
        parts.append("it belongs to the same TMDB collection")
    elif components and components.get("tmdb_similar", 0.0) >= 1.0:
        parts.append("TMDB lists it as a similar title to your anchor")
    elif components and components.get("tmdb_recommendation", 0.0) >= 1.0:
        parts.append("TMDB recommends it as a related follow-up title")
    if components and components.get("canonical_theme_boost", 0.0) >= 0.9:
        parts.append("it is a widely recognized example of this concept")
    if components and components.get("multi_theme_strength"):
        mts = str(components.get("multi_theme_strength"))
        if mts == "strong":
            parts.append("this is a **strong direct thematic match** (animal-centered and performance/musical-focused)")
        elif mts == "partial":
            parts.append("this is a **closer thematic match** with clear animal and some music/performance elements")
    if components and components.get("franchise_note"):
        parts.append(str(components.get("franchise_note")))
    elif components and float(components.get("franchise_score", 0.0)) > 0.5:
        parts.append("This appears to be a continuation or related title in the same series.")
    if shared_genres:
        parts.append("it shares " + " and ".join([f"**{g}**" for g in shared_genres[:3]]) + " elements")
    if shared_kw:
        parts.append("overlapping themes like " + ", ".join([f"`{k}`" for k in shared_kw[:4]]))
    parts.append(f"a similar emotional tone (sentiment match ~{tone*100:.0f}%)")

    if components and components.get("genre_overlap", 0) >= 2:
        parts.append("with a stronger multi-genre overlap boost")
    if components and components.get("query_metadata_boost", 0) >= 0.34:
        parts.append("and strong metadata alignment with your query terms")
    if components and components.get("centrality_score", 0) >= 0.66:
        parts.append("with strong concept centrality in title/keywords/overview")
    elif components and components.get("centrality_score", 0) >= 0.45:
        parts.append("with moderate concept centrality")

    return f"**{title}** {prefix} " + ", ".join(parts) + "."


def genre_only_recommendations(
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 18,
    requested_genre_ids: set[int] | None = None,
    keyword_hint: str | None = None,
) -> list[tuple[dict[str, Any], float, dict[str, float]]]:
    """
    When the user only provides genres/filters (no query, no anchor), rank using
    a conservative popularity+rating signal. Relevance is enforced by hard filters upstream.
    """
    if not candidates:
        return []
    max_votes = float(max((m.get("vote_count") or 0) for m in candidates) or 1.0)
    scored: list[tuple[dict[str, Any], float, dict[str, float]]] = []
    kh = (keyword_hint or "").strip().lower()
    for m in candidates:
        pop = _popularity_component(m, max_votes)
        gids = _genre_ids(m)
        if requested_genre_ids:
            overlap = len(gids & requested_genre_ids)
            if overlap <= 0:
                continue
            gscore = compute_genre_overlap_score(overlap)
        else:
            gscore = 0.5
        kw_score = 0.0
        if kh:
            kws = " ".join(extract_keyword_names(m)).lower()
            if kh in kws:
                kw_score = 1.0
        score = 0.55 * gscore + 0.35 * pop + 0.10 * kw_score
        comps = {
            "content": 0.0,
            "sentiment": 0.5,
            "genre_overlap_score": float(gscore),
            "popularity": float(pop),
            "keyword_hint_score": float(kw_score),
            "hybrid": float(score),
        }
        scored.append((m, float(score), comps))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(1, int(top_n))]


def build_result_list(
    exact_ranked: list[tuple[dict[str, Any], float, dict[str, float]]],
    closest_ranked: list[tuple[dict[str, Any], float, dict[str, float]]],
    *,
    target_n: int = 10,
) -> list[tuple[dict[str, Any], float, dict[str, float]]]:
    """
    Build a final multi-title list with dedupe by movie id, prioritizing exact matches
    and then filling with closest valid matches until target_n is reached.
    """
    out: list[tuple[dict[str, Any], float, dict[str, float]]] = []
    seen: set[int] = set()
    for src, tag in ((exact_ranked, None), (closest_ranked, "closest")):
        for m, s, c in src:
            mid = m.get("id")
            if not isinstance(mid, int):
                continue
            if mid in seen:
                continue
            seen.add(mid)
            cc = dict(c or {})
            if tag and not cc.get("match_level"):
                cc["match_level"] = tag
            out.append((m, s, cc))
            if len(out) >= max(1, int(target_n)):
                return out
    return out


def hybrid_with_fallback(
    anchor: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 15,
    mood_sentiment_target: float | None = None,
    content_sim_threshold: float = DEFAULT_CONTENT_SIM_THRESHOLD,
    fallback_content_sim_threshold: float = FALLBACK_CONTENT_SIM_THRESHOLD,
    w_content: float = W_CONTENT,
    w_sentiment: float = W_SENTIMENT,
    w_genre: float = W_GENRE,
    w_popularity: float = W_POPULARITY,
    relationship_context: dict[str, Any] | None = None,
) -> tuple[list[tuple[dict[str, Any], float, dict[str, float]]], bool]:
    """
    Fallback wrapper:
    - first pass: DEFAULT_CONTENT_SIM_THRESHOLD
    - fallback pass: FALLBACK_CONTENT_SIM_THRESHOLD (still requires >=1 genre overlap)
    Returns (ranked, is_fallback).
    """
    ranked = hybrid_recommendations(
        anchor,
        candidates,
        top_n=top_n,
        mood_sentiment_target=mood_sentiment_target,
        content_sim_threshold=content_sim_threshold,
        w_content=w_content,
        w_sentiment=w_sentiment,
        w_genre=w_genre,
        w_popularity=w_popularity,
        relationship_context=relationship_context,
    )
    if ranked:
        return ranked, False
    ranked = hybrid_recommendations(
        anchor,
        candidates,
        top_n=top_n,
        mood_sentiment_target=mood_sentiment_target,
        content_sim_threshold=fallback_content_sim_threshold,
        w_content=w_content,
        w_sentiment=w_sentiment,
        w_genre=w_genre,
        w_popularity=w_popularity,
        relationship_context=relationship_context,
    )
    return ranked, True


def query_with_fallback(
    query_text: str,
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 18,
    mood_sentiment_target: float | None = None,
    requested_genre_ids: set[int] | None = None,
) -> tuple[list[tuple[dict[str, Any], float, dict[str, float]]], bool]:
    """
    Fallback wrapper for query-based ranking:
    - first pass: DEFAULT_CONTENT_SIM_THRESHOLD
    - fallback: FALLBACK_CONTENT_SIM_THRESHOLD
    Keeps genre-overlap requirement when requested_genre_ids is provided.
    """
    ranked = query_recommendations(
        query_text,
        candidates,
        top_n=top_n,
        mood_sentiment_target=mood_sentiment_target,
        requested_genre_ids=requested_genre_ids,
        content_sim_threshold=DEFAULT_CONTENT_SIM_THRESHOLD,
    )
    if ranked:
        return ranked, False
    ranked = query_recommendations(
        query_text,
        candidates,
        top_n=top_n,
        mood_sentiment_target=mood_sentiment_target,
        requested_genre_ids=requested_genre_ids,
        content_sim_threshold=FALLBACK_CONTENT_SIM_THRESHOLD,
    )
    return ranked, True
