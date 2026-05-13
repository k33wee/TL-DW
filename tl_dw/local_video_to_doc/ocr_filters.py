from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Sequence

from .models import VisualNote


MEANINGFUL_VISUAL_HINTS = {
    "meet",
    "meeting",
    "participant",
    "participants",
    "recording",
    "present",
    "presenting",
    "screen",
    "shared",
    "captions",
    "agenda",
    "speaker",
    "host",
    "joined",
    "left",
    "chat",
    "camera",
    "microphone",
    "mute",
    "raise hand",
}


def normalize_ocr_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" |\t\r\n")


def is_meaningful_ocr_text(text: str, min_chars: int) -> bool:
    normalized = normalize_ocr_text(text)
    if not normalized:
        return False

    lowered = normalized.lower()
    if lowered in {"google meet", "meet"}:
        return True
    if any(hint in lowered for hint in MEANINGFUL_VISUAL_HINTS):
        return True
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", normalized):
        return False
    if len(normalized) < min_chars:
        return False
    return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", normalized))


def dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = normalize_ocr_text(value)
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def filter_meaningful_ocr_lines(
    raw_lines: Sequence[tuple[str, float]],
    min_score: float,
    min_chars: int,
) -> list[tuple[str, float]]:
    filtered: list[tuple[str, float]] = []
    seen: set[str] = set()
    for text, score in raw_lines:
        cleaned = normalize_ocr_text(text)
        key = cleaned.casefold()
        if score < min_score or key in seen:
            continue
        if not is_meaningful_ocr_text(cleaned, min_chars):
            continue
        seen.add(key)
        filtered.append((cleaned, score))
    return filtered


def summarize_visual_lines(lines: Sequence[str], max_lines: int, max_chars: int) -> str:
    kept_lines = dedupe_preserve_order(lines)[:max_lines]
    if not kept_lines:
        return ""

    parts: list[str] = []
    total = 0
    for line in kept_lines:
        addition = len(line) + (3 if parts else 0)
        if parts and total + addition > max_chars:
            break
        parts.append(line)
        total += addition

    summary = " | ".join(parts)
    return summary[:max_chars].rstrip(" |")


def visual_note_similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=left.casefold(), b=right.casefold()).ratio()


def dedupe_visual_notes(
    notes: Sequence[VisualNote],
    dedupe_window_sec: float,
    similarity_threshold: float = 0.92,
) -> list[VisualNote]:
    deduped: list[VisualNote] = []
    for note in notes:
        if not deduped:
            deduped.append(note)
            continue

        previous = deduped[-1]
        within_window = note.timestamp - previous.timestamp <= dedupe_window_sec
        if (
            within_window
            and visual_note_similarity(previous.text, note.text) >= similarity_threshold
        ):
            continue
        deduped.append(note)
    return deduped
