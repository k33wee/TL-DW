from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from tl_dw.common.paths import DEFAULT_OUTPUT_DIR, VIDEO_EXTENSIONS
from tl_dw.common.process import run_command

from .models import MediaChapter, MediaInfo, OutputPaths
from .text_utils import normalize_whitespace, safe_slug


def ffprobe_metadata(video_path: Path) -> dict[str, Any]:
    raw = run_command(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            str(video_path),
        ]
    )
    return json.loads(raw)


def extract_source_chapters(
    raw_chapters: Sequence[dict[str, Any]] | None,
    duration: float,
) -> list[MediaChapter]:
    chapters: list[MediaChapter] = []
    for index, raw_chapter in enumerate(raw_chapters or [], start=1):
        try:
            start = float(raw_chapter.get("start_time") or 0.0)
            end = float(raw_chapter.get("end_time") or duration or 0.0)
        except (TypeError, ValueError):
            continue

        if duration > 0:
            start = max(0.0, min(start, duration))
            end = max(0.0, min(end, duration))
        if end <= start:
            continue

        tags = raw_chapter.get("tags") or {}
        fallback_title = f"Chapter {index}"
        title = normalize_whitespace(str(tags.get("title") or fallback_title))
        chapters.append(
            MediaChapter(title=title or fallback_title, start=start, end=end)
        )

    chapters.sort(key=lambda chapter: chapter.start)
    return chapters


def media_info_from_metadata(video_path: Path, metadata: dict[str, Any]) -> MediaInfo:
    format_info = metadata.get("format") or {}
    tags = format_info.get("tags") or {}

    try:
        duration = float(format_info.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    raw_title = str(tags.get("title") or video_path.stem.replace("_", " "))
    title = normalize_whitespace(raw_title) or video_path.stem
    chapters = extract_source_chapters(metadata.get("chapters"), duration)
    return MediaInfo(
        title=title, duration=duration, chapters=chapters, source_path=video_path
    )


def extract_audio(video_path: Path, wav_out: Path) -> None:
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(wav_out),
        ]
    )


def extract_audio_chunk(
    audio_path: Path, chunk_out: Path, start: float, duration: float
) -> None:
    run_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(audio_path),
            "-acodec",
            "pcm_s16le",
            str(chunk_out),
        ]
    )


def build_initial_prompt(
    title: str,
    chapters: Sequence[MediaChapter],
    override: str | None = None,
) -> str | None:
    if override:
        cleaned = normalize_whitespace(override)
        return cleaned or None

    parts = [normalize_whitespace(title)]
    chapter_titles = [
        normalize_whitespace(chapter.title)
        for chapter in chapters
        if normalize_whitespace(chapter.title)
        and normalize_whitespace(chapter.title).lower() != "transcript"
    ]
    if chapter_titles:
        parts.append(". ".join(chapter_titles[:8]))

    prompt = ". ".join(part for part in parts if part)
    return prompt or None


def choose_transcription_chapters(
    media_info: MediaInfo,
    ignore_source_chapters: bool,
) -> tuple[list[MediaChapter], bool]:
    if media_info.chapters and not ignore_source_chapters:
        return media_info.chapters, True

    end = media_info.duration if media_info.duration > 0 else 0.0
    return [MediaChapter(title="Transcript", start=0.0, end=end)], False


def resolve_output_paths(
    video_path: Path,
    output_target: Path | None,
    multiple_videos: bool,
) -> OutputPaths:
    slug = safe_slug(video_path.stem)
    if output_target is None:
        artifact_dir = DEFAULT_OUTPUT_DIR / slug
        document_path = artifact_dir / "document.md"
    elif multiple_videos:
        if output_target.suffix.lower() == ".md":
            raise SystemExit(
                "When processing multiple videos, --output must be a directory, not a Markdown file."
            )
        artifact_dir = output_target / slug
        document_path = artifact_dir / "document.md"
    elif output_target.suffix.lower() == ".md":
        artifact_dir = output_target.parent / f"{slug}_artifacts"
        document_path = output_target
    else:
        artifact_dir = output_target / slug
        document_path = artifact_dir / "document.md"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    document_path.parent.mkdir(parents=True, exist_ok=True)
    return OutputPaths(artifact_dir=artifact_dir, document_path=document_path)


def resolve_videos(video_path: Path | None, video_dir: Path) -> list[Path]:
    if video_path is not None:
        candidate = video_path.expanduser().resolve()
        if not candidate.exists():
            raise SystemExit(f"Video file not found: {candidate}")
        if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
            raise SystemExit(f"Unsupported video extension for {candidate.name}")
        return [candidate]

    candidate_dir = video_dir.expanduser().resolve()
    if not candidate_dir.exists() or not candidate_dir.is_dir():
        raise SystemExit(f"Video directory not found: {candidate_dir}")

    videos = sorted(
        path
        for path in candidate_dir.iterdir()
        if path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise SystemExit(f"No supported video files found in {candidate_dir}")
    return videos
