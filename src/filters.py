"""
Declarative movie filters (year, rating, votes, runtime, certification, language,
cast/crew, genres with AND/OR semantics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Literal

from src import api

GenreMode = Literal["AND", "OR"]


@dataclass
class MovieFilters:
    year_min: int | None = None
    year_max: int | None = None
    min_rating: float | None = None
    min_vote_count: int | None = None
    runtime_min: int | None = None
    runtime_max: int | None = None
    certification: str | None = None  # e.g. PG-13, R — matched against US cert when possible
    language: str | None = None  # ISO 639-1, e.g. en
    actor_names: list[str] = field(default_factory=list)
    director_names: list[str] = field(default_factory=list)
    genre_ids: list[int] = field(default_factory=list)
    genre_mode: GenreMode = "OR"
    enforce_runtime: bool = True  # if False, runtime constraints are treated as soft/default
    year_active: bool = False
    vote_count_active: bool = False
    studio_contains: str | None = None
    country_contains: str | None = None
    keyword_contains: str | None = None
    exclude_animation: bool = False
    exclude_documentary: bool = False
    exclude_horror: bool = False
    exclude_adult: bool = False
    # Query-derived exclusion (e.g. "no superheroes", "without animation")
    exclude_superhero: bool = False
    excluded_genre_ids: list[int] = field(default_factory=list)
    audience_level: str | None = None  # Kids / Teen / Mature
    sort_by: str = "Best match"
    allow_uncertified_in_fallback: bool = False

UNSAFE_KEYWORDS_KIDS: set[str] = {
    "kidnapping",
    "murder",
    "serial killer",
    "rape",
    "abduction",
    "torture",
    "basement",
    "porn",
    "sex",
    "drug",
    "cartel",
    "terrorism",
    "assassin",
}

GENRE_ID_ANIMATION = 16
GENRE_ID_COMEDY = 35
GENRE_ID_CRIME = 80
GENRE_ID_FAMILY = 10751
GENRE_ID_FANTASY = 14
GENRE_ID_HORROR = 27
GENRE_ID_MYSTERY = 9648
GENRE_ID_THRILLER = 53
GENRE_ID_WAR = 10752
GENRE_ID_ADVENTURE = 12

logger = logging.getLogger(__name__)

SUPERHERO_THEME_FRAGMENTS: tuple[str, ...] = (
    "superhero",
    "super power",
    "superpower",
    "comic",
    "comic book",
    "vigilante",
    "masked",
    "marvel",
    "dc",
    "dc comics",
    "avengers",
)


def superhero_theme_triggers(movie: dict[str, Any]) -> list[str]:
    """
    Return the strongest superhero-theme triggers found in the hydrated movie record.
    Uses keywords + collection name + title/overview hints (best-effort).
    """
    triggers: list[str] = []
    title_l = str(movie.get("title") or "").lower()
    overview_l = str(movie.get("overview") or "").lower()

    # Collection / franchise name
    bc = movie.get("belongs_to_collection")
    if isinstance(bc, dict):
        name_l = str(bc.get("name") or "").lower()
        if any(x in name_l for x in ("marvel", "dc")):
            triggers.append("collection")

    # TMDB keywords
    kw_block = movie.get("keywords") or {}
    kw_list = kw_block.get("keywords") if isinstance(kw_block, dict) else None
    if isinstance(kw_list, list):
        kw_names = [str(k.get("name") or "").lower() for k in kw_list if isinstance(k, dict)]
        for kw in kw_names:
            if any(frag in kw for frag in SUPERHERO_THEME_FRAGMENTS):
                if "hero" in kw and not any(x in kw for x in ("comic", "vigilante", "masked", "super")):
                    continue
                triggers.append("keyword")
                break

    # Title/overview
    if any(x in title_l for x in ("marvel", "dc", "avengers", "superhero")):
        triggers.append("title")
    if any(x in overview_l for x in ("superhero", "comic book", "vigilante", "masked", "superpower", "super power")):
        triggers.append("overview")

    out: list[str] = []
    for t in triggers:
        if t not in out:
            out.append(t)
    return out


def matches_superhero_theme(movie: dict[str, Any]) -> bool:
    """Hard gate for superhero-themed queries: requires at least one strong trigger."""
    return len(superhero_theme_triggers(movie)) > 0


DC_FRANCHISE_SIGNALS: tuple[str, ...] = (
    "dc",
    "dc comics",
    "gotham",
    "batman",
    "superman",
    "wonder woman",
    "aquaman",
    "the flash",
    "flash",
    "green lantern",
    "justice league",
    "joker",
    "harley quinn",
    "shazam",
    "black adam",
    "suicide squad",
    "teen titans",
    "birds of prey",
)

MARVEL_FRANCHISE_SIGNALS: tuple[str, ...] = (
    "marvel",
    "avengers",
    "iron man",
    "captain america",
    "thor",
    "hulk",
    "black widow",
    "spider-man",
    "spiderman",
    "guardians",
    "guardians of the galaxy",
    "x-men",
    "doctor strange",
    "wanda",
    "deadpool",
    "fantastic four",
    "ant-man",
    "loki",
    "black panther",
)

ANIMAL_THEME_SIGNALS: tuple[str, ...] = (
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


def _franchise_text(movie: dict[str, Any]) -> str:
    title_l = str(movie.get("title") or "").lower()
    overview_l = str(movie.get("overview") or "").lower()
    bc = movie.get("belongs_to_collection")
    col_l = str(bc.get("name") or "").lower() if isinstance(bc, dict) else ""
    kw_block = movie.get("keywords") or {}
    kw_list = kw_block.get("keywords") if isinstance(kw_block, dict) else None
    kw_l = ""
    if isinstance(kw_list, list):
        kw_l = " ".join([str(k.get("name") or "").lower() for k in kw_list if isinstance(k, dict)])
    return " ".join([title_l, overview_l, col_l, kw_l]).strip()


def matches_franchise(movie: dict[str, Any], required_franchise: str) -> bool:
    """
    Hard franchise constraint:
    - required="dc": must match DC signals and must NOT match Marvel signals
    - required="marvel": must match Marvel signals and must NOT match DC signals
    """
    req = (required_franchise or "").strip().lower()
    if req not in {"dc", "marvel"}:
        return True
    text = _franchise_text(movie)
    dc_hit = any(s in text for s in DC_FRANCHISE_SIGNALS)
    mv_hit = any(s in text for s in MARVEL_FRANCHISE_SIGNALS)
    if req == "dc":
        return bool(dc_hit) and not bool(mv_hit)
    return bool(mv_hit) and not bool(dc_hit)


def matches_animal_theme(movie: dict[str, Any]) -> bool:
    text = _franchise_text(movie)
    return any(s in text for s in ANIMAL_THEME_SIGNALS)


def exclusion_penalty_for_intent(
    movie: dict[str, Any],
    *,
    excluded_genre_ids: list[int] | None,
    exclude_superhero: bool,
) -> float:
    """
    Reusable exclusion penalty helper for generalized ranking models.
    Returns >=1.0 when movie violates explicit exclusion intent.
    """
    penalty = 0.0
    if exclude_superhero and matches_superhero_theme(movie):
        penalty += 1.0
    if excluded_genre_ids:
        want = {int(g) for g in excluded_genre_ids if isinstance(g, int)}
        if _movie_genre_ids(movie) & want:
            penalty += 1.0
    return penalty


def _movie_genre_ids_set(movie: dict[str, Any]) -> set[int]:
    return {int(gid) for gid in _movie_genre_ids(movie) if isinstance(gid, int)}


def _is_superhero_related(movie: dict[str, Any]) -> bool:
    """
    Best-effort superhero/Marvel/DC exclusion detector.
    Primarily uses keywords + belongs_to_collection and title hints.
    """
    # Strong explicit signals
    title_l = str(movie.get("title") or "").lower()
    if any(x in title_l for x in ["marvel", "dc", "avengers"]):
        return True

    bc = movie.get("belongs_to_collection")
    if isinstance(bc, dict):
        name_l = str(bc.get("name") or "").lower()
        if any(x in name_l for x in ["marvel", "dc"]):
            return True

    # Keywords (most reliable for this use-case)
    kw_block = movie.get("keywords") or {}
    kw_list = kw_block.get("keywords") if isinstance(kw_block, dict) else None
    if isinstance(kw_list, list):
        kw_names = [str(k.get("name") or "").lower() for k in kw_list if isinstance(k, dict)]
        superhero_fragments = [
            "superhero",
            "superpower",
            "comic",
            "vigilante",
            "masked",
            "hero",
        ]
        for kw in kw_names:
            if any(frag in kw for frag in superhero_fragments):
                # reduce false positives: if we only matched generic "hero", require more context
                if "hero" in kw and not any(x in kw for x in ["comic", "vigilante", "masked", "super"]):
                    continue
                return True

    # Last resort: overview hints (keep it strict).
    # Avoid matching common words like "marvel" in unrelated contexts.
    overview_l = str(movie.get("overview") or "").lower()
    if any(x in overview_l for x in ["superhero", "comic book", "vigilante", "masked"]):
        return True

    return False


def _norm_list(xs: list[str] | None) -> list[str]:
    if not xs:
        return []
    return [x.strip().lower() for x in xs if x and x.strip()]


def _movie_genre_ids(movie: dict[str, Any]) -> set[int]:
    return {g.get("id") for g in movie.get("genres", []) if g.get("id")}


def _movie_year(movie: dict[str, Any]) -> int | None:
    rd = movie.get("release_date") or ""
    if rd and len(rd) >= 4:
        try:
            return int(rd[:4])
        except ValueError:
            return None
    return None


def _contains_any(haystack: list[str], needle_fragments: list[str]) -> bool:
    if not needle_fragments:
        return True
    hs = [x.lower() for x in haystack if x]
    return all(any(n in h for h in hs) for n in needle_fragments)


def genre_overlap_count(movie: dict[str, Any], genre_ids: list[int]) -> int:
    mids = _movie_genre_ids(movie)
    want = {int(g) for g in (genre_ids or [])}
    return len(mids & want)


def _passes_genres(movie: dict[str, Any], f: MovieFilters, *, relaxed_and: bool = False) -> bool:
    if not f.genre_ids:
        return True
    mids = _movie_genre_ids(movie)
    if f.genre_mode == "AND":
        if relaxed_and:
            # Keep partial matches; scoring/ranking can boost full matches later.
            return any(g in mids for g in f.genre_ids)
        return all(g in mids for g in f.genre_ids)
    return any(g in mids for g in f.genre_ids)


def _passes_cast_crew(movie: dict[str, Any], f: MovieFilters) -> bool:
    credits = movie.get("credits")
    actors_need = _norm_list(f.actor_names)
    directors_need = _norm_list(f.director_names)

    if not actors_need and not directors_need:
        return True

    if not credits:
        return False

    cast_names = {c.get("name", "").lower() for c in credits.get("cast", []) if c.get("name")}
    director_names = {n.lower() for n in api.extract_directors(credits) if n}

    for frag in actors_need:
        if not any(frag in cn for cn in cast_names):
            return False

    for frag in directors_need:
        if not any(frag in dn for dn in director_names):
            return False

    return True


def _passes_studio_country_keyword(movie: dict[str, Any], f: MovieFilters) -> bool:
    if f.studio_contains:
        studios = [c.get("name", "") for c in (movie.get("production_companies") or []) if isinstance(c, dict)]
        if not _contains_any(studios, [f.studio_contains.strip().lower()]):
            return False
    if f.country_contains:
        countries = [c.get("name", "") for c in (movie.get("production_countries") or []) if isinstance(c, dict)]
        countries += [c.get("iso_3166_1", "") for c in (movie.get("production_countries") or []) if isinstance(c, dict)]
        if not _contains_any(countries, [f.country_contains.strip().lower()]):
            return False
    if f.keyword_contains:
        kw_block = movie.get("keywords") or {}
        kws = [k.get("name", "") for k in (kw_block.get("keywords") or []) if isinstance(k, dict)]
        if not _contains_any(kws, [f.keyword_contains.strip().lower()]):
            return False
    return True


def _passes_excludes(movie: dict[str, Any], f: MovieFilters) -> bool:
    return not violates_exclusions(movie, f)


def violates_exclusions(movie: dict[str, Any], f: MovieFilters) -> bool:
    """
    Absolute exclusion gate. If True, movie must be removed before scoring.
    Checks genre names/ids, adult flag, and keyword hints when available.
    """
    gids = _movie_genre_ids(movie)
    gnames = {str(g.get("name", "")).strip().lower() for g in (movie.get("genres") or []) if isinstance(g, dict)}

    # Exclude any genres explicitly requested to be excluded by the query parser.
    if f.excluded_genre_ids:
        want = {int(gid) for gid in f.excluded_genre_ids if isinstance(gid, int)}
        if gids & want:
            return True

    if f.exclude_horror and (GENRE_ID_HORROR in gids or "horror" in gnames):
        return True
    if f.exclude_documentary and (99 in gids or "documentary" in gnames):
        return True
    if f.exclude_animation and (GENRE_ID_ANIMATION in gids or "animation" in gnames):
        return True

    if f.exclude_adult:
        if bool(movie.get("adult")):
            return True
        # Best-effort adult-content keyword screening when available
        kw_block = movie.get("keywords") or {}
        kws = [str(k.get("name", "")).strip().lower() for k in (kw_block.get("keywords") or []) if isinstance(k, dict)]
        adult_like = ("porn", "sexual", "erotic", "nudity", "sex", "adult")
        if any(any(tag in kw for tag in adult_like) for kw in kws):
            return True

    if f.exclude_superhero and _is_superhero_related(movie):
        return True
    return False


def apply_hard_exclusions(
    movies: list[dict[str, Any]],
    f: MovieFilters,
    *,
    debug_context: str = "",
) -> list[dict[str, Any]]:
    """
    Remove excluded content prior to scoring and explanations.
    This is intentionally separate from generic filtering so exclusion rules
    remain absolute even in fallback/closest-match modes.
    """
    if not movies:
        return []
    kept = [m for m in movies if not violates_exclusions(m, f)]
    removed = len(movies) - len(kept)
    if removed > 0:
        logger.debug(
            "Hard exclusions removed %s/%s candidates (%s) [superhero=%s excluded_genres=%s horror=%s doc=%s anim=%s adult=%s]",
            removed,
            len(movies),
            debug_context or "unknown",
            f.exclude_superhero,
            len(getattr(f, "excluded_genre_ids", []) or []),
            f.exclude_horror,
            f.exclude_documentary,
            f.exclude_animation,
            f.exclude_adult,
        )
    return kept


def _passes_audience_level(movie: dict[str, Any], f: MovieFilters) -> bool:
    if not f.audience_level:
        return True
    level = f.audience_level.lower()
    cert = (api.us_certification_from_details(movie) or "").upper()
    gids = _movie_genre_ids(movie)

    # Always block explicit adult flag for Kids/Teen (safety-first).
    if level in {"kids / family", "teen"} and bool(movie.get("adult")):
        return False

    if level == "mature":
        return True

    if level == "teen":
        # Teen: allow PG/PG-13; allow R only if user explicitly asked for R certification.
        if cert in {"PG", "PG-13"}:
            return True
        if cert == "R":
            return (f.certification or "").upper() == "R"
        # If no cert data, allow (will still be constrained by other hard filters).
        return cert == ""

    if level == "kids / family":
        # Kids/Family: disallow scary/dark-heavy genres unless strong family balancing exists.
        if GENRE_ID_HORROR in gids or GENRE_ID_WAR in gids:
            return False

        # Crime/thriller are usually unsafe for kids unless balanced by family/animation.
        has_family_signal = (GENRE_ID_FAMILY in gids) or (GENRE_ID_ANIMATION in gids)
        if (GENRE_ID_CRIME in gids or GENRE_ID_THRILLER in gids) and not has_family_signal:
            return False

        # Keyword-based safety check (best-effort).
        kw_block = movie.get("keywords") or {}
        kws = [str(k.get("name", "")).lower() for k in (kw_block.get("keywords") or []) if isinstance(k, dict)]
        if any(any(bad in kw for bad in UNSAFE_KEYWORDS_KIDS) for kw in kws):
            return False

        # Certification logic:
        # - G/PG are allowed
        # - PG-13 is only allowed if it has strong family signals and isn't crime/thriller heavy
        # - missing cert is allowed only if fallback explicitly permits it
        if cert in {"G", "PG"}:
            return True
        if cert == "PG-13":
            if not has_family_signal:
                return False
            if GENRE_ID_CRIME in gids or GENRE_ID_THRILLER in gids:
                return False
            return True
        if cert == "":
            return bool(f.allow_uncertified_in_fallback) and has_family_signal
        return False

    return True


def detect_filter_conflicts(f: MovieFilters) -> list[str]:
    """Detect contradictory safety + genre/cert combinations before running recommendations."""
    issues: list[str] = []
    lvl = (f.audience_level or "").lower()
    cert = (f.certification or "").upper()
    selected = set(int(x) for x in (f.genre_ids or []))
    if lvl == "kids / family":
        if cert in {"R", "NC-17"}:
            issues.append("Kids / Family conflicts with certification R/NC-17.")
        if GENRE_ID_HORROR in selected:
            issues.append("Kids / Family rarely overlaps with Horror.")
        if GENRE_ID_CRIME in selected:
            issues.append("Kids / Family rarely overlaps with Crime.")
        if GENRE_ID_WAR in selected:
            issues.append("Kids / Family rarely overlaps with War.")
        if GENRE_ID_THRILLER in selected:
            issues.append("Kids / Family rarely overlaps with Thriller.")
        if f.keyword_contains and any(bad in f.keyword_contains.lower() for bad in UNSAFE_KEYWORDS_KIDS):
            issues.append("Kids / Family conflicts with dark/adult keyword intent.")
    return issues


def movie_passes_filters(
    movie: dict[str, Any],
    f: MovieFilters,
    *,
    relaxed_genres_for_and: bool = False,
) -> bool:
    """
    Returns True if full movie detail dict satisfies all active constraints.
    Expects genres, runtime, vote_average, vote_count, credits when cast/director filters used.
    """
    if f.year_min is not None or f.year_max is not None:
        if f.year_active:
            y = _movie_year(movie)
            if y is None:
                return False
            if f.year_min is not None and y < f.year_min:
                return False
            if f.year_max is not None and y > f.year_max:
                return False

    if f.min_rating is not None:
        va = movie.get("vote_average")
        if va is None or float(va) < float(f.min_rating):
            return False

    if f.min_vote_count is not None and f.vote_count_active:
        vc = movie.get("vote_count") or 0
        if int(vc) < int(f.min_vote_count):
            return False

    rt = movie.get("runtime")
    if f.enforce_runtime and (f.runtime_min is not None or f.runtime_max is not None):
        if rt is None or int(rt) <= 0:
            return False
        r = int(rt)
        if f.runtime_min is not None and r < f.runtime_min:
            return False
        if f.runtime_max is not None and r > f.runtime_max:
            return False

    if f.certification:
        cert = api.us_certification_from_details(movie)
        want = f.certification.strip().lower()
        if not cert:
            # In fallback mode, allow missing cert only (never other certs).
            if not f.allow_uncertified_in_fallback:
                return False
        elif cert.strip().lower() != want:
            return False

    if f.language:
        lang = (movie.get("original_language") or "").lower()
        if lang != f.language.strip().lower():
            return False

    if not _passes_genres(movie, f, relaxed_and=relaxed_genres_for_and):
        return False

    if not _passes_cast_crew(movie, f):
        return False

    if not _passes_studio_country_keyword(movie, f):
        return False

    if not _passes_excludes(movie, f):
        return False

    if not _passes_audience_level(movie, f):
        return False

    return True


def apply_filters(
    movies: list[dict[str, Any]],
    f: MovieFilters,
    *,
    relaxed_genres_for_and: bool = False,
) -> list[dict[str, Any]]:
    return [m for m in movies if movie_passes_filters(m, f, relaxed_genres_for_and=relaxed_genres_for_and)]


def merge_parsed_nl_into_filters(base: MovieFilters, parsed: Any) -> MovieFilters:
    """Overlay ParsedQuery from nlp_utils onto an existing MovieFilters instance."""
    if getattr(parsed, "genre_ids", None):
        base.genre_ids = list({*base.genre_ids, *parsed.genre_ids})
    if getattr(parsed, "year_min", None) is not None:
        base.year_min = (
            parsed.year_min if base.year_min is None else max(base.year_min, parsed.year_min)
        )
    if getattr(parsed, "year_max", None) is not None:
        base.year_max = (
            parsed.year_max if base.year_max is None else min(base.year_max, parsed.year_max)
        )
    if getattr(parsed, "runtime_max", None) is not None:
        base.runtime_max = (
            parsed.runtime_max
            if base.runtime_max is None
            else min(base.runtime_max, parsed.runtime_max)
        )
    if getattr(parsed, "runtime_min", None) is not None:
        base.runtime_min = (
            parsed.runtime_min
            if base.runtime_min is None
            else max(base.runtime_min, parsed.runtime_min)
        )
    return base
