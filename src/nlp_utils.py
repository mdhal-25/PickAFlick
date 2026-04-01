"""
Sentiment analysis and natural-language filter parsing for PickAFlick.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from textblob import TextBlob

# TMDB genre name -> id (common set for parsing)
GENRE_NAME_TO_ID: dict[str, int] = {
    "action": 28,
    "adventure": 12,
    "animation": 16,
    "comedy": 35,
    "crime": 80,
    "documentary": 99,
    "drama": 18,
    "family": 10751,
    "fantasy": 14,
    "history": 36,
    "horror": 27,
    "music": 10402,
    "mystery": 9648,
    "romance": 10749,
    "science fiction": 878,
    "sci-fi": 878,
    "scifi": 878,
    "tv movie": 10770,
    "thriller": 53,
    "war": 10752,
    "western": 37,
}

KEYWORD_GENRE_HINTS: dict[str, list[int]] = {
    # romance
    "romantic": [GENRE_NAME_TO_ID["romance"]],
    "love": [GENRE_NAME_TO_ID["romance"]],
    "romance": [GENRE_NAME_TO_ID["romance"]],
    # comedy
    "funny": [GENRE_NAME_TO_ID["comedy"]],
    "comedy": [GENRE_NAME_TO_ID["comedy"]],
    # thriller / drama
    "dark": [GENRE_NAME_TO_ID["thriller"], GENRE_NAME_TO_ID["drama"]],
    "thrilling": [GENRE_NAME_TO_ID["thriller"]],
    "lethal": [GENRE_NAME_TO_ID["thriller"], GENRE_NAME_TO_ID["action"]],
    # horror
    "scary": [GENRE_NAME_TO_ID["horror"]],
    "horror": [GENRE_NAME_TO_ID["horror"]],
    # action/adventure/family/animation
    "action": [GENRE_NAME_TO_ID["action"]],
    "adventure": [GENRE_NAME_TO_ID["adventure"]],
    "family": [GENRE_NAME_TO_ID["family"]],
    "animated": [GENRE_NAME_TO_ID["animation"]],
    "cartoon": [GENRE_NAME_TO_ID["animation"]],
    "animation": [GENRE_NAME_TO_ID["animation"]],
    # sci-fi
    "sci-fi": [GENRE_NAME_TO_ID["science fiction"]],
    "scifi": [GENRE_NAME_TO_ID["science fiction"]],
    "science fiction": [GENRE_NAME_TO_ID["science fiction"]],
    # other
    "mystery": [GENRE_NAME_TO_ID["mystery"]],
    "crime": [GENRE_NAME_TO_ID["crime"]],
    "war": [GENRE_NAME_TO_ID["war"]],
    "fantasy": [GENRE_NAME_TO_ID["fantasy"]],
    "emotional": [GENRE_NAME_TO_ID["drama"]],
    "sad": [GENRE_NAME_TO_ID["drama"]],
    "drama": [GENRE_NAME_TO_ID["drama"]],
}

GENERIC_WEAK_WORDS: set[str] = {"film", "movie", "story", "life", "world"}


@dataclass
class ParsedQuery:
    """Structured output from informal user text."""

    genre_ids: list[int] = field(default_factory=list)
    genre_names: list[str] = field(default_factory=list)
    excluded_genre_ids: list[int] = field(default_factory=list)
    excluded_keywords: list[str] = field(default_factory=list)
    exclude_superhero: bool = False
    year_min: int | None = None
    year_max: int | None = None
    runtime_max: int | None = None
    runtime_min: int | None = None
    mood: str | None = None  # positive, negative, neutral hint for sentiment matching
    require_superhero_theme: bool = False
    require_animal_theme: bool = False
    require_music_theme: bool = False
    required_franchise: str | None = None  # "dc" | "marvel"
    raw: str = ""


@dataclass
class QueryIntent:
    """Generalized query intent used by ranking model."""

    explicit_genre_ids: list[int] = field(default_factory=list)
    themes: dict[str, list[str]] = field(default_factory=dict)
    mood: str | None = None
    required_franchise: str | None = None
    excluded_genre_ids: list[int] = field(default_factory=list)
    exclude_superhero: bool = False
    raw_terms: list[str] = field(default_factory=list)
    raw_text: str = ""


SUPERHERO_THEME_TERMS: tuple[str, ...] = (
    "superhero",
    "super hero",
    "superpower",
    "super power",
    "comic",
    "comic book",
    "vigilante",
    "masked",
    "marvel",
    "dc",
    "dc comics",
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

THEME_BUCKETS: dict[str, tuple[str, ...]] = {
    "superhero": SUPERHERO_THEME_TERMS,
    "animal": ANIMAL_THEME_TERMS,
    "music": MUSIC_THEME_TERMS,
    "family": ("family", "kids", "child", "children"),
    "mystery": ("mystery", "detective", "investigation"),
}


def _parse_required_themes(text: str, parsed: ParsedQuery) -> None:
    t = (text or "").lower()
    if any(term in t for term in SUPERHERO_THEME_TERMS):
        parsed.require_superhero_theme = True
    if any(term in t for term in ANIMAL_THEME_TERMS):
        parsed.require_animal_theme = True
    if any(term in t for term in MUSIC_THEME_TERMS):
        parsed.require_music_theme = True


def _parse_required_franchise(text: str, parsed: ParsedQuery) -> None:
    t = (text or "").lower()
    # Use word-boundary regexes so short queries like "DC" / "Marvel" work.
    dc_hit = bool(
        re.search(
            r"\b(dc|dc universe|dc comics|justice league|batman universe)\b",
            t,
        )
    )
    mv_hit = bool(
        re.search(
            r"\b(marvel|marvel universe|mcu|avengers|x-men|spider-man|spiderman|guardians of the galaxy)\b",
            t,
        )
    )
    if dc_hit and not mv_hit:
        parsed.required_franchise = "dc"
        parsed.require_superhero_theme = True
    elif mv_hit and not dc_hit:
        parsed.required_franchise = "marvel"
        parsed.require_superhero_theme = True


def overview_sentiment(text: str | None) -> float:
    """
    Polarity for movie overview in [-1, 1] using TextBlob.
    Returns 0.0 for empty/missing text.
    """
    if not text or not str(text).strip():
        return 0.0
    return float(TextBlob(str(text)).sentiment.polarity)


def sentiment_similarity(score_a: float, score_b: float) -> float:
    """
    Similarity in [0, 1]: 1 = same polarity magnitude and direction.
    Uses absolute difference on the [-1,1] scale.
    """
    sa = max(-1.0, min(1.0, float(score_a)))
    sb = max(-1.0, min(1.0, float(score_b)))
    dist = abs(sa - sb)  # 0 .. 2
    return 1.0 - (dist / 2.0)


def _parse_runtime_phrases(text: str, parsed: ParsedQuery) -> None:
    t = text.lower()
    m = re.search(r"under\s+(\d+)\s*(?:hour|hours|hr|hrs)\b", t)
    if m:
        parsed.runtime_max = int(m.group(1)) * 60
    m = re.search(r"less than\s+(\d+)\s*(?:hour|hours|hr|hrs)\b", t)
    if m:
        parsed.runtime_max = int(m.group(1)) * 60
    m = re.search(r"(?:over|more than|at least)\s+(\d+)\s*(?:hour|hours|hr|hrs)\b", t)
    if m:
        parsed.runtime_min = int(m.group(1)) * 60
    m = re.search(r"under\s+(\d+)\s*minutes?\b", t)
    if m:
        parsed.runtime_max = int(m.group(1))
    m = re.search(r"(?:over|more than)\s+(\d+)\s*minutes?\b", t)
    if m:
        parsed.runtime_min = int(m.group(1))


def _parse_decade(text: str, parsed: ParsedQuery) -> None:
    t = text.lower()
    # 2000s / 2010s / 1990s
    m = re.search(r"\b(18|19|20)(\d{2})0s\b", t)
    if m:
        base = int(m.group(1) + m.group(2) + "0")
        parsed.year_min = base
        parsed.year_max = base + 9
        return
    m = re.search(r"\b(18|19|20)(\d{2})s\b", t)
    if m:
        base = int(m.group(1) + m.group(2))
        # If they wrote 1990s-style without the explicit 0, treat as decade start.
        parsed.year_min = base
        parsed.year_max = base + 9
        return
    # 90s / 80s / 00s / 10s (assume 1900s for >=30, else 2000s)
    m = re.search(r"\b(\d{2})s\b", t)
    if m:
        two = int(m.group(1))
        start = (1900 + two) if two >= 30 else (2000 + two)
        parsed.year_min = start
        parsed.year_max = start + 9
        return


def _parse_mood(text: str, parsed: ParsedQuery) -> None:
    t = text.lower()
    positive = ("funny", "lighthearted", "uplifting", "feel-good", "feel good", "heartwarming")
    negative = ("dark", "gritty", "bleak", "depressing", "sad", "disturbing")
    if any(w in t for w in positive):
        parsed.mood = "positive"
    elif any(w in t for w in negative):
        parsed.mood = "negative"


def _parse_genres(text: str, parsed: ParsedQuery) -> None:
    t = text.lower()
    for name, gid in GENRE_NAME_TO_ID.items():
        if re.search(rf"\b{re.escape(name)}\b", t):
            if gid not in parsed.genre_ids:
                parsed.genre_ids.append(gid)
                parsed.genre_names.append(name)
    # keyword-to-genre multi-intent hints (romantic, lethal, dark, etc.)
    for kw, gids in KEYWORD_GENRE_HINTS.items():
        if re.search(rf"\b{re.escape(kw)}\b", t):
            for gid in gids:
                if gid not in parsed.genre_ids:
                    parsed.genre_ids.append(gid)
                    # best-effort reverse name
                    for n, ngid in GENRE_NAME_TO_ID.items():
                        if ngid == gid:
                            if n not in parsed.genre_names:
                                parsed.genre_names.append(n)
                            break


def parse_natural_language_query(user_text: str) -> ParsedQuery:
    """
    Convert informal phrases into structured hints (genres, years, runtime, mood).
    Example: "funny action movie under 2 hours from the 2000s"
    """
    parsed = ParsedQuery(raw=user_text or "")
    if not user_text or not user_text.strip():
        return parsed

    original = user_text.strip()

    # 1) Extract negative preferences first (so they don't get re-interpreted as positives).
    # We remove negative phrases only for genre/mood parsing, but we parse runtime/decade from the original.
    cleaned_for_positive = _strip_negative_phrases(original, parsed)

    # 2) Parse positive genres/mood from cleaned text.
    _parse_genres(cleaned_for_positive, parsed)
    _parse_mood(cleaned_for_positive, parsed)
    _parse_required_themes(original, parsed)
    _parse_required_franchise(original, parsed)

    # 3) Parse time constraints from the original query text.
    _parse_runtime_phrases(original, parsed)
    _parse_decade(original, parsed)
    return parsed


def parse_query_intent(user_text: str, parsed: ParsedQuery | None = None) -> QueryIntent:
    """
    General decomposition for ranking:
    - explicit genres
    - theme buckets
    - mood
    - exclusions
    - franchise constraints
    """
    p = parsed or parse_natural_language_query(user_text or "")
    txt = (user_text or "").lower()
    terms = [t for t in re.split(r"\s+", txt) if t]

    themes: dict[str, list[str]] = {}
    for name, bucket_terms in THEME_BUCKETS.items():
        hits = [t for t in bucket_terms if t in txt]
        if hits:
            themes[name] = hits

    return QueryIntent(
        explicit_genre_ids=list(p.genre_ids),
        themes=themes,
        mood=p.mood,
        required_franchise=p.required_franchise,
        excluded_genre_ids=list(p.excluded_genre_ids),
        exclude_superhero=bool(p.exclude_superhero),
        raw_terms=terms,
        raw_text=user_text or "",
    )


def _strip_negative_phrases(text: str, parsed: ParsedQuery) -> str:
    """
    Detect negative preference phrases and populate `parsed` with exclusions.
    Returns a cleaned text where those negative phrases are removed, so genre/mood parsing
    doesn't treat negated terms as positives.
    """
    # stop at punctuation / end; keep it simple to avoid long captures
    # also supports: "I don't want X", "dont want X"
    neg_re = re.compile(r"(?i)\b(?:no|not|without|exclude|anything but|dont want|don't want)\s+([^,.;!?]+)")
    matches = list(neg_re.finditer(text))
    if not matches:
        return text

    # Build a cleaned string by blanking the matched spans (avoid index drift).
    spans: list[tuple[int, int]] = [(m.start(), m.end()) for m in matches]
    spans.sort(key=lambda x: x[0])
    cleaned_parts: list[str] = []
    prev = 0
    for s, e in spans:
        cleaned_parts.append(text[prev:s])
        cleaned_parts.append(" ")
        prev = e
    cleaned_parts.append(text[prev:])
    cleaned = "".join(cleaned_parts)

    def _genre_name_variants(name: str) -> list[str]:
        n = name.lower().strip()
        if not n:
            return []
        if n.endswith("y"):
            return [n, n[:-1] + "ies"]
        return [n, n + "s"]

    def _segment_has_genre(seg_l: str, genre_name: str) -> bool:
        # Exact token/word match or plural variant match.
        for v in _genre_name_variants(genre_name):
            if re.search(rf"\b{re.escape(v)}\b", seg_l):
                return True
        # fallback: startswith (e.g. "romance-driven")
        return seg_l.startswith(genre_name)

    for m in matches:
        segment = m.group(1).strip()
        if not segment:
            continue
        seg_l = segment.lower()

        # Superhero / Marvel / DC exclusions
        if any(k in seg_l for k in ["superhero", "super power", "superpower", "marvel", "dc", "comic book", "vigilante", "masked"]):
            parsed.exclude_superhero = True

        # Excluded theme keywords
        if any(k in seg_l for k in ["marvel", "dc"]):
            parsed.excluded_keywords.append("marvel/dc")

        # Excluded genre mapping
        excluded: set[int] = set()
        # direct genre names (support basic plural)
        for name, gid in GENRE_NAME_TO_ID.items():
            if _segment_has_genre(seg_l, name):
                excluded.add(gid)
        # keyword hints mapping (romantic, lethal, horror, etc.)
        for kw, gids in KEYWORD_GENRE_HINTS.items():
            if re.search(rf"\b{re.escape(kw)}\b", seg_l):
                for gid in gids:
                    excluded.add(gid)

        for gid in excluded:
            if gid not in parsed.excluded_genre_ids:
                parsed.excluded_genre_ids.append(gid)

    return cleaned


def has_positive_recommendation_signal(parsed: ParsedQuery) -> bool:
    """
    Returns True if query has any positive signal we can use for retrieval.
    Negative-only requests should be blocked by the caller.
    """
    return bool(parsed.genre_ids) or bool(parsed.mood)


def is_short_structured_genre_query(
    raw_text: str,
    parsed: ParsedQuery,
    query_terms: list[str],
) -> bool:
    """
    Detect short, genre-style queries where text-semantic similarity is often weak
    but structured genre retrieval is strong (e.g. "romantic action", "crime comedy").
    """
    if not raw_text or not raw_text.strip():
        return False
    if len(query_terms) == 0 or len(query_terms) > 6:
        return False
    if not (parsed.genre_ids or parsed.mood):
        return False

    genre_words: set[str] = set()
    for g in parsed.genre_names:
        for t in re.split(r"\s+", g.lower().strip()):
            if t:
                genre_words.add(t)
    # Include common query terms seen in this app.
    genre_words.update({"romantic", "action", "crime", "comedy", "family", "adventure", "thriller", "horror", "drama", "mystery", "fantasy", "animation"})
    if parsed.mood:
        genre_words.add(parsed.mood.lower())

    hits = sum(1 for t in query_terms if t.lower() in genre_words)
    return (hits / max(1, len(query_terms))) >= 0.6


def is_negative_only_request(parsed: ParsedQuery, *, has_any_genres_selected: bool, has_anchor: bool) -> bool:
    """
    Negative-only means:
    - we detected exclusions
    - and there are no positive genre/mood signals from text
    - and sidebar genre selections are empty
    - and no anchor movie is selected
    """
    has_exclusions = bool(parsed.exclude_superhero) or bool(parsed.excluded_genre_ids) or bool(parsed.excluded_keywords)
    if not has_exclusions:
        return False
    if has_any_genres_selected:
        return False
    if has_anchor:
        return False
    return not has_positive_recommendation_signal(parsed)


def query_has_genre_or_mood(parsed: ParsedQuery) -> bool:
    return bool(parsed.genre_ids) or bool(parsed.mood)


def extract_query_terms(user_text: str) -> list[str]:
    """
    Lightweight term extraction for query-based matching (no heavy NLP deps).
    Removes very generic weak words and returns distinct lowercased tokens.
    """
    if not user_text or not user_text.strip():
        return []
    t = user_text.lower()
    t = re.sub(r"[^a-z0-9\s\-]", " ", t)
    tokens = [w for w in re.split(r"\s+", t) if w]
    out: list[str] = []
    seen: set[str] = set()
    for w in tokens:
        if w in GENERIC_WEAK_WORDS:
            continue
        if len(w) <= 2:
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def parsed_mood_target_sentiment(mood: str | None) -> float | None:
    """Optional anchor sentiment for hybrid mood alignment."""
    if mood == "positive":
        return 0.35
    if mood == "negative":
        return -0.35
    return None
