from __future__ import annotations

import re
from typing import Any, Sequence

from .models import Paragraph, RenderedChapter, TranscriptChapter, VisualNote
from .paragraphs import segment_paragraphs
from .text_utils import normalize_whitespace


def generate_chapter_title(paragraphs: Sequence[Paragraph], index: int) -> str:
    if not paragraphs:
        return f"Part {index}"

    candidate = re.split(r"(?<=[.!?])\s+", paragraphs[0].text.strip(), maxsplit=1)[0]
    candidate = normalize_whitespace(candidate).strip(" -–—:;,.()[]{}")
    if len(candidate) > 64:
        candidate = candidate[:64].rsplit(" ", 1)[0].rstrip(" -–—:;,.()[]{}")
    if len(candidate) < 8:
        return f"Part {index}"
    return f"Part {index} — {candidate}"


def split_paragraphs_into_generated_chapters(
    paragraphs: Sequence[Paragraph],
    max_chapter_seconds: float,
    min_gap_seconds: float,
) -> list[RenderedChapter]:
    if not paragraphs:
        return []

    groups: list[list[Paragraph]] = []
    current_group = [paragraphs[0]]
    current_group_start = paragraphs[0].start

    for paragraph in paragraphs[1:]:
        gap = paragraph.start - current_group[-1].start
        current_duration = paragraph.start - current_group_start
        should_split = current_duration >= max_chapter_seconds or (
            gap >= min_gap_seconds and current_duration >= max_chapter_seconds / 2
        )
        if should_split:
            groups.append(current_group)
            current_group = [paragraph]
            current_group_start = paragraph.start
        else:
            current_group.append(paragraph)

    groups.append(current_group)
    return [
        RenderedChapter(
            title=generate_chapter_title(group, index + 1),
            start=group[0].start,
            paragraphs=list(group),
        )
        for index, group in enumerate(groups)
    ]


def build_document_chapters(
    transcript_chapters: Sequence[TranscriptChapter],
    sat_model: Any,
    use_source_chapters: bool,
    segment_unchaptered: bool,
    max_generated_chapter_seconds: float,
    min_generated_chapter_gap: float,
) -> list[RenderedChapter]:
    if use_source_chapters:
        return _build_source_chapters(transcript_chapters, sat_model)

    all_segments = [
        segment for chapter in transcript_chapters for segment in chapter.segments
    ]
    paragraphs = segment_paragraphs(all_segments, sat_model)
    if not paragraphs:
        return []
    if segment_unchaptered:
        return split_paragraphs_into_generated_chapters(
            paragraphs=paragraphs,
            max_chapter_seconds=max_generated_chapter_seconds,
            min_gap_seconds=min_generated_chapter_gap,
        )
    return [
        RenderedChapter(
            title="Transcript", start=paragraphs[0].start, paragraphs=paragraphs
        )
    ]


def attach_visual_notes_to_chapters(
    chapters: Sequence[RenderedChapter],
    visual_notes: Sequence[VisualNote],
) -> list[RenderedChapter]:
    if not chapters:
        return _build_visual_only_chapters(visual_notes)

    augmented = [
        RenderedChapter(
            title=chapter.title,
            start=chapter.start,
            paragraphs=list(chapter.paragraphs),
            visual_notes=[],
        )
        for chapter in chapters
    ]
    if not visual_notes:
        return augmented

    chapter_index = 0
    for note in visual_notes:
        while (
            chapter_index + 1 < len(augmented)
            and note.timestamp >= augmented[chapter_index + 1].start
        ):
            chapter_index += 1
        augmented[chapter_index].visual_notes.append(note)
    return augmented


def _build_source_chapters(
    transcript_chapters: Sequence[TranscriptChapter],
    sat_model: Any,
) -> list[RenderedChapter]:
    rendered: list[RenderedChapter] = []
    for chapter in transcript_chapters:
        paragraphs = segment_paragraphs(chapter.segments, sat_model)
        if paragraphs:
            rendered.append(
                RenderedChapter(
                    title=chapter.title,
                    start=paragraphs[0].start,
                    paragraphs=paragraphs,
                )
            )
    return rendered


def _build_visual_only_chapters(
    visual_notes: Sequence[VisualNote],
) -> list[RenderedChapter]:
    if not visual_notes:
        return []
    return [
        RenderedChapter(
            title="Visual context",
            start=visual_notes[0].timestamp,
            paragraphs=[],
            visual_notes=list(visual_notes),
        )
    ]
