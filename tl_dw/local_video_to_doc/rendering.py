from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from tl_dw.common.time_utils import sec_to_hhmmss

from .models import RenderedChapter
from .text_utils import safe_slug


def build_unique_anchors(chapters: Sequence[RenderedChapter]) -> list[str]:
    seen: dict[str, int] = {}
    anchors: list[str] = []
    for chapter in chapters:
        base = safe_slug(chapter.title) or "chapter"
        occurrence = seen.get(base, 0) + 1
        seen[base] = occurrence
        anchors.append(base if occurrence == 1 else f"{base}-{occurrence}")
    return anchors


def render_markdown(
    title: str,
    source_path: Path,
    chapters: Sequence[RenderedChapter],
    timestamp_paragraphs: bool,
    add_table_of_contents: bool,
) -> str:
    lines = [f"# {title}", "", f"Source: `{source_path}`"]
    if not chapters:
        lines.extend(["", "No transcript content was generated."])
        return "\n".join(lines)

    anchors = build_unique_anchors(chapters)
    if add_table_of_contents and len(chapters) > 1:
        lines.extend(["", "### Table of contents"])
        for chapter, anchor in zip(chapters, anchors, strict=True):
            lines.append(
                f"- {sec_to_hhmmss(chapter.start)} [{chapter.title}](#{anchor})"
            )

    for chapter, anchor in zip(chapters, anchors, strict=True):
        lines.extend(["", f'## {chapter.title} <a name="{anchor}"></a>'])
        if chapter.visual_notes:
            lines.extend(["", "### Visual context"])
            for note in chapter.visual_notes:
                lines.append(f"- {sec_to_hhmmss(note.timestamp)}: {note.text}")
        for paragraph in chapter.paragraphs:
            prefix = (
                f"({sec_to_hhmmss(paragraph.start)}) " if timestamp_paragraphs else ""
            )
            lines.extend(["", f"{prefix}{paragraph.text}"])

    return "\n".join(lines)


def save_rendered_chapters(
    chapters: Sequence[RenderedChapter], output_dir: Path
) -> None:
    payload = [
        {
            "title": chapter.title,
            "start": chapter.start,
            "paragraphs": [
                {"start": paragraph.start, "text": paragraph.text}
                for paragraph in chapter.paragraphs
            ],
            "visual_notes": [
                {
                    "timestamp": note.timestamp,
                    "text": note.text,
                    "lines": note.lines,
                    "confidence": note.confidence,
                }
                for note in chapter.visual_notes
            ],
        }
        for chapter in chapters
    ]
    output_path = output_dir / "chapters.json"
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
