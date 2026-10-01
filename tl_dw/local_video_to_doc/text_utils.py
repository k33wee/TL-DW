from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Sequence


_TITLE_STOPWORDS = frozenset(
    """
    a about above after again against all also am an and another any are as at
    be because been before being below between both but by can could
    da dal dalla dalle dei del della delle dello dell di do does doing down during
    each ed eh ehm for from further gli ha hai hanno he her here hers herself
    him himself his how i if in into is it its itself il la le li lo loro lui
    ma me mi mia mie mio no non not now of off on once only or other our ours
    ourselves out over own per pero piu poi same she should so some such than
    that the their theirs them themselves then there these they this those
    through to too tra un una uno under until up very was we were what when
    where which while who whom why will with would ye yeah yes yet you your
    yours yourself yourselves allora anche ancora bene buongiorno buonasera
    ciao comunque cosa cosi dunque ecco essere grazie insomma niente okay ok
    praticamente proprio quindi questo quella quelle quello questi questa
    queste salve siamo siete sono stato stata tipo vabbe vediamo verso oggi
    parliamo appunto senti scusa scusate hello hi well thanks thank
    cioe questa questi fare fatto faccio fatta problema problemi capire capito
    secondo abbiamo solo due punto punti gia modo modi sia parte parti roba
    cose tutto tutti tutte qualche vorrei voglio potrebbe dovrebbe adesso
    magari qualcosa detto viene vanno facciamo bisogna motivo domanda sempre
    proprio ora qui qua molto troppo senza dopo prima dentro fuori sopra sotto
    altro altra altri altre stesso stessa stessi stesse nostro nostra perche
    quando dove come adesso allora appunto diciamo vedere visto guarda guardare
    che con nel nella negli nelle sul sui sullo sulla sulle dei degli dalle
    dalle dagli dai col coi nei negli agli alle allo alla ai al fra dati
    cazzo cavolo merda porco madonna minchia vaffanculo boiate boiata cazzata
    cazzate figa stronzo stronzi
    """.split()
)


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_segment_text(value: str) -> str:
    return re.sub(r"\s+", " ", value)


def safe_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value).strip("-").lower()
    return slug or "document"


def build_summary_title(
    texts: Sequence[str],
    fallback: str,
    *,
    max_chars: int = 90,
) -> str:
    """Build a short topic title from recurring transcript terms."""
    raw_words = [
        word
        for text in texts
        for word in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    ]
    folded = [_fold_token(word) for word in raw_words]
    counted = Counter(
        token for token in folded if len(token) >= 4 and token not in _TITLE_STOPWORDS
    )
    if not counted:
        return _fallback_title(fallback)

    lemmas = {token: _lemma(token, counted) for token in counted}
    tokens = [
        (index, raw_words[index], lemmas[folded[index]])
        for index, token in enumerate(folded)
        if token in lemmas
    ]
    lemma_counts: Counter[str] = Counter(lemma for _, _, lemma in tokens)
    first_at = {lemma: index for index, _, lemma in reversed(tokens)}
    surfaces: dict[str, Counter[str]] = {}
    for _, raw, lemma in tokens:
        surfaces.setdefault(lemma, Counter())[_surface(raw)] += 1

    total = sum(lemma_counts.values())
    phrase_min = 3 if total >= 80 else 2
    word_min = 4 if total >= 80 else 2
    phrase_counts: Counter[tuple[str, str]] = Counter()
    phrase_at: dict[tuple[str, str], int] = {}
    for (left_at, _, left), (right_at, _, right) in zip(tokens, tokens[1:], strict=False):
        if right_at != left_at + 1 or left == right:
            continue
        key = (left, right)
        phrase_counts[key] += 1
        phrase_at.setdefault(key, left_at)

    phrases = [
        key
        for key, count in phrase_counts.items()
        if count >= phrase_min
    ]
    phrases.sort(key=lambda key: (-phrase_counts[key], phrase_at[key]))
    chosen: list[tuple[int, str]] = []
    used: set[str] = set()
    for key in phrases:
        if any(part in used for part in key):
            continue
        chosen.append((phrase_at[key], _phrase_label(key, surfaces)))
        used.update(key)
        break

    words = [
        lemma
        for lemma, count in lemma_counts.items()
        if lemma not in used and count >= word_min
    ]
    words.sort(key=lambda lemma: (-lemma_counts[lemma], first_at[lemma]))
    for lemma in words:
        if len(chosen) >= 4:
            break
        chosen.append((first_at[lemma], _surface_label(surfaces[lemma])))

    if len(chosen) < 2:
        return _fallback_title(fallback)
    chosen.sort(key=lambda item: item[0])
    return _trim_title(_join_title([label for _, label in chosen]), max_chars)


def _fallback_title(fallback: str) -> str:
    return normalize_whitespace(fallback) or "Meeting"


def _lemma(token: str, counts: Counter[str]) -> str:
    if len(token) > 4 and token.endswith("e"):
        singular = token[:-1] + "a"
        if counts[singular] >= 2:
            return singular
    if len(token) > 4 and token.endswith("i"):
        singular = token[:-1] + "o"
        if counts[singular] >= 2:
            return singular
    if token.endswith("ing") and len(token) > 7 and counts[token[:-3]] >= 2:
        return token[:-3]
    return token


def _phrase_label(key: tuple[str, str], surfaces: dict[str, Counter[str]]) -> str:
    return " ".join(_surface_label(surfaces[part]) for part in key)


def _surface(raw: str) -> str:
    letters = [character for character in raw if character.isalpha()]
    if letters and sum(character.isupper() for character in letters) / len(letters) > 0.8:
        return raw.upper()
    return raw.lower()


def _surface_label(surfaces: Counter[str]) -> str:
    return surfaces.most_common(1)[0][0]


def _join_title(labels: Sequence[str]) -> str:
    kept = [label for label in labels if label]
    if len(kept) == 1:
        title = kept[0]
    elif len(kept) == 2:
        title = f"{kept[0]} e {kept[1]}"
    else:
        title = f"{', '.join(kept[:-1])} e {kept[-1]}"
    return title[:1].upper() + title[1:]


def _fold_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def _trim_title(value: str, max_chars: int) -> str:
    cleaned = normalize_whitespace(value).strip(" -:;,.!?\"'")
    if len(cleaned) <= max_chars:
        return cleaned
    shortened = cleaned[:max_chars].rsplit(" ", 1)[0].strip(" -:;,.!?\"'")
    return shortened or cleaned[:max_chars].strip()
