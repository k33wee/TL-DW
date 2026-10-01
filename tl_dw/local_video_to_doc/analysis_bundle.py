from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from tl_dw.common.time_utils import sec_to_hhmmss

from .models import FrameObservation, MediaInfo, TranscriptChapter


ANALYSIS_SCHEMA_VERSION = 1


def build_analysis_sections(
    transcript_chapters: Sequence[TranscriptChapter],
    frames: Sequence[FrameObservation],
    *,
    media_duration: float,
    section_seconds: float,
) -> list[dict[str, Any]]:
    """Build bounded, chronological units suitable for multimodal model calls."""
    if section_seconds <= 0:
        raise ValueError("section_seconds must be greater than 0")

    latest_transcript_end = max(
        (
            segment.end
            for chapter in transcript_chapters
            for segment in chapter.segments
        ),
        default=0.0,
    )
    latest_frame = max((frame.timestamp for frame in frames), default=0.0)
    effective_duration = max(media_duration, latest_transcript_end, latest_frame)
    section_count = max(1, math.ceil(max(effective_duration, 0.001) / section_seconds))

    sections: list[dict[str, Any]] = []
    for index in range(section_count):
        start = index * section_seconds
        end = (
            effective_duration
            if index == section_count - 1
            else min(effective_duration, (index + 1) * section_seconds)
        )
        sections.append(
            {
                "id": f"section-{index + 1:03d}",
                "index": index + 1,
                "start": round(start, 3),
                "end": round(max(start, end), 3),
                "transcript": [],
                "transcript_text": "",
                "frames": [],
            }
        )

    transcript_rows: list[list[dict[str, Any]]] = [[] for _ in sections]
    for chapter in transcript_chapters:
        for segment in chapter.segments:
            midpoint = segment.start + max(0.0, segment.end - segment.start) / 2
            section_index = _section_index(
                midpoint, section_seconds=section_seconds, section_count=section_count
            )
            transcript_rows[section_index].append(
                {
                    "chapter": chapter.title,
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment.text.strip(),
                }
            )

    for frame in frames:
        section_index = _section_index(
            frame.timestamp,
            section_seconds=section_seconds,
            section_count=section_count,
        )
        sections[section_index]["frames"].append(
            {
                "timestamp": round(frame.timestamp, 3),
                "image_path": frame.image_path,
                "change_score": frame.change_score,
                "reasons": frame.reasons,
                "ocr_text": frame.ocr_text,
                "ocr_lines": frame.ocr_lines,
                "ocr_confidence": frame.ocr_confidence,
            }
        )

    for section, rows in zip(sections, transcript_rows, strict=True):
        rows.sort(key=lambda row: (row["start"], row["end"]))
        section["transcript"] = rows
        section["transcript_text"] = "\n".join(
            f"[{sec_to_hhmmss(row['start'])} - {sec_to_hhmmss(row['end'])}] "
            f"{row['text']}"
            for row in rows
        )
        section["frames"].sort(key=lambda frame: frame["timestamp"])

    return sections


def write_analysis_bundle(
    output_path: Path,
    *,
    media_info: MediaInfo,
    transcript_chapters: Sequence[TranscriptChapter],
    frames: Sequence[FrameObservation],
    section_seconds: float,
    settings: dict[str, Any],
    document_path: Path | None = None,
) -> dict[str, Any]:
    sections = build_analysis_sections(
        transcript_chapters,
        frames,
        media_duration=media_info.duration,
        section_seconds=section_seconds,
    )
    transcript_segment_count = sum(
        len(chapter.segments) for chapter in transcript_chapters
    )
    payload: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "kind": "tl-dw-meeting-analysis",
        "title": media_info.title,
        "source": {
            "path": str(media_info.source_path),
            "duration": media_info.duration,
        },
        "artifacts": {
            "transcript_text": "transcript.txt",
            "transcript_jsonl": "transcript.jsonl",
            "frames": "frames.json",
            "document": (
                Path(os.path.relpath(document_path, output_path.parent)).as_posix()
                if document_path is not None
                else "document.md"
            ),
        },
        "settings": settings,
        "stats": {
            "sections": len(sections),
            "transcript_segments": transcript_segment_count,
            "selected_frames": len(frames),
        },
        "sections": sections,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def _section_index(
    timestamp: float, *, section_seconds: float, section_count: int
) -> int:
    return min(section_count - 1, max(0, int(max(0.0, timestamp) // section_seconds)))
