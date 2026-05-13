from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from tl_dw.common.console import log
from tl_dw.common.time_utils import sec_to_hhmmss

from .media import extract_audio_chunk
from .models import MediaChapter, TranscriptChapter, TranscriptSegment
from .text_utils import normalize_segment_text


def load_whisper_model(model_name: str, device: str, compute_type: str) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'faster-whisper'. Install extraction dependencies before running this module."
        ) from exc

    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except ValueError as exc:
        if device != "cuda":
            raise
        return _retry_cuda_model_load(
            whisper_model_cls=WhisperModel,
            model_name=model_name,
            requested_compute_type=compute_type,
            original_error=exc,
        )


def _retry_cuda_model_load(
    whisper_model_cls: Any,
    model_name: str,
    requested_compute_type: str,
    original_error: ValueError,
) -> Any:
    fallback_types = [
        compute_type
        for compute_type in ["int8", "float16", "default"]
        if compute_type != requested_compute_type
    ]
    for compute_type in fallback_types:
        try:
            log(
                "Requested CUDA compute type was not supported, "
                f"retrying with '{compute_type}'."
            )
            return whisper_model_cls(
                model_name, device="cuda", compute_type=compute_type
            )
        except ValueError:
            continue
    raise original_error


def transcribe_media(
    audio_path: Path,
    media_duration: float,
    chapters: Sequence[MediaChapter],
    whisper_model: Any,
    requested_language: str | None,
    initial_prompt: str | None,
    verbose_logger: callable,
) -> tuple[list[TranscriptChapter], dict[str, Any]]:
    transcript_chapters: list[TranscriptChapter] = []
    detected_language = requested_language
    chapter_info: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="local-video-doc-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for index, chapter in enumerate(chapters, start=1):
            chapter_duration = max(0.0, chapter.end - chapter.start)
            if chapter_duration <= 0.0 and media_duration > 0:
                continue

            use_full_audio = (
                len(chapters) == 1
                and chapter.start <= 0.001
                and (media_duration <= 0.0 or abs(chapter.end - media_duration) <= 0.5)
            )

            chapter_audio_path = audio_path
            if not use_full_audio:
                chapter_audio_path = temp_dir / f"chapter_{index:03d}.wav"
                extract_audio_chunk(
                    audio_path, chapter_audio_path, chapter.start, chapter_duration
                )

            kwargs: dict[str, Any] = {"beam_size": 5, "vad_filter": True}
            runtime_language = requested_language or detected_language
            if runtime_language:
                kwargs["language"] = runtime_language
            if initial_prompt:
                kwargs["initial_prompt"] = initial_prompt

            verbose_logger(f"Transcribing chapter {index}: {chapter.title}")
            segments_iter, info = whisper_model.transcribe(
                str(chapter_audio_path), **kwargs
            )

            chapter_segments: list[TranscriptSegment] = []
            for segment in segments_iter:
                text = normalize_segment_text(segment.text or "")
                if not text.strip():
                    continue
                chapter_segments.append(
                    TranscriptSegment(
                        start=float(segment.start) + chapter.start,
                        end=float(segment.end) + chapter.start,
                        text=text,
                    )
                )

            info_language = getattr(info, "language", None)
            if detected_language is None and info_language:
                detected_language = str(info_language)

            chapter_info.append(
                {
                    "title": chapter.title,
                    "start": chapter.start,
                    "end": chapter.end,
                    "segments": len(chapter_segments),
                    "detected_language": info_language,
                }
            )
            transcript_chapters.append(
                TranscriptChapter(title=chapter.title, segments=chapter_segments)
            )

    return transcript_chapters, {
        "requested_language": requested_language,
        "detected_language": detected_language,
        "initial_prompt": initial_prompt,
        "chapters": chapter_info,
    }


def save_transcript(
    transcript_chapters: Sequence[TranscriptChapter], output_dir: Path
) -> None:
    transcript_lines: list[str] = []
    jsonl_path = output_dir / "transcript.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for chapter in transcript_chapters:
            for segment in chapter.segments:
                clean_text = segment.text.strip()
                transcript_lines.append(
                    f"[{sec_to_hhmmss(segment.start)} - {sec_to_hhmmss(segment.end)}] "
                    f"[{chapter.title}] {clean_text}"
                )
                jsonl_file.write(
                    json.dumps(
                        {
                            "chapter": chapter.title,
                            "start": segment.start,
                            "end": segment.end,
                            "text": clean_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    transcript_path = output_dir / "transcript.txt"
    transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
