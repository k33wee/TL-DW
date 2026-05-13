from __future__ import annotations

import json
from argparse import Namespace
from typing import Any

from tl_dw.common.console import log, vlog

from .chapters import attach_visual_notes_to_chapters, build_document_chapters
from .media import (
    build_initial_prompt,
    choose_transcription_chapters,
    extract_audio,
    ffprobe_metadata,
    media_info_from_metadata,
    resolve_output_paths,
)
from .models import VisualNote
from .ocr import extract_visual_notes
from .rendering import render_markdown, save_rendered_chapters
from .transcription import save_transcript, transcribe_media


def process_video(
    video_path,
    args: Namespace,
    whisper_model: Any,
    sat_model: Any,
    ocr_engine: Any | None,
    multiple_videos: bool,
) -> None:
    log(f"\n=== Processing: {video_path.name} ===")
    output_paths = resolve_output_paths(video_path, args.output, multiple_videos)
    vlog(args.verbose, f"Artifact directory: {output_paths.artifact_dir}")

    metadata = ffprobe_metadata(video_path)
    media_info = media_info_from_metadata(video_path, metadata)
    _write_json(output_paths.artifact_dir / "metadata.json", metadata)

    visual_notes = _extract_visual_context(
        video_path=video_path,
        media_duration=media_info.duration,
        args=args,
        ocr_engine=ocr_engine,
        artifact_dir=output_paths.artifact_dir,
    )

    audio_path = output_paths.artifact_dir / "audio.wav"
    log("  - Extracting audio")
    extract_audio(video_path, audio_path)

    transcription_chapters, use_source_chapters = choose_transcription_chapters(
        media_info=media_info,
        ignore_source_chapters=args.ignore_source_chapters,
    )
    initial_prompt = build_initial_prompt(
        media_info.title,
        transcription_chapters,
        args.initial_prompt,
    )

    log("  - Transcribing speech")
    transcript_chapters, transcript_info = transcribe_media(
        audio_path=audio_path,
        media_duration=media_info.duration,
        chapters=transcription_chapters,
        whisper_model=whisper_model,
        requested_language=args.whisper_language,
        initial_prompt=initial_prompt,
        verbose_logger=lambda message: vlog(args.verbose, message),
    )
    save_transcript(transcript_chapters, output_paths.artifact_dir)

    transcript_info.update(
        {
            "whisper_model": args.whisper_model,
            "whisper_device": args.whisper_device,
            "whisper_compute_type": args.whisper_compute_type,
            "sat_model": args.sat_model,
            "used_source_chapters": use_source_chapters,
            "segment_unchaptered": args.segment_unchaptered,
            "use_ocr": args.use_ocr,
            "ocr_sample_sec": args.ocr_sample_sec,
            "ocr_notes": len(visual_notes),
        }
    )
    _write_json(output_paths.artifact_dir / "transcript_info.json", transcript_info)

    log("  - Segmenting into readable paragraphs")
    rendered_chapters = build_document_chapters(
        transcript_chapters=transcript_chapters,
        sat_model=sat_model,
        use_source_chapters=use_source_chapters,
        segment_unchaptered=args.segment_unchaptered,
        max_generated_chapter_seconds=args.max_generated_chapter_seconds,
        min_generated_chapter_gap=args.min_generated_chapter_gap,
    )
    rendered_chapters = attach_visual_notes_to_chapters(rendered_chapters, visual_notes)
    save_rendered_chapters(rendered_chapters, output_paths.artifact_dir)

    markdown = render_markdown(
        title=media_info.title,
        source_path=media_info.source_path,
        chapters=rendered_chapters,
        timestamp_paragraphs=args.timestamp_paragraphs,
        add_table_of_contents=args.add_table_of_contents,
    )
    output_paths.document_path.write_text(markdown, encoding="utf-8")
    log(f"  - Wrote Markdown: {output_paths.document_path}")


def _extract_visual_context(
    video_path, media_duration, args, ocr_engine, artifact_dir
) -> list[VisualNote]:
    if not args.use_ocr or ocr_engine is None:
        return []

    log("  - Extracting meaningful on-screen text")
    return extract_visual_notes(
        video_path=video_path,
        media_duration=media_duration,
        ocr_engine=ocr_engine,
        sample_sec=args.ocr_sample_sec,
        min_score=args.ocr_min_score,
        min_chars=args.ocr_min_chars,
        max_lines=args.ocr_max_lines,
        max_note_chars=args.ocr_max_note_chars,
        dedupe_window_sec=args.ocr_dedupe_window_sec,
        artifact_dir=artifact_dir,
        verbose_logger=lambda message: vlog(args.verbose, message),
    )


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
