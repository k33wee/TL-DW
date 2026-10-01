from __future__ import annotations

import json
from argparse import Namespace
from typing import Any

from tl_dw.common.console import log, vlog

from .analysis_bundle import write_analysis_bundle
from .chapters import attach_visual_notes_to_chapters, build_document_chapters
from .frames import extract_meaningful_frames
from .media import (
    build_initial_prompt,
    choose_transcription_chapters,
    extract_audio,
    ffprobe_metadata,
    media_info_from_metadata,
    resolve_output_paths,
)
from .models import FrameObservation, VisualNote
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

    audio_path = output_paths.artifact_dir / "audio.wav"
    normalization_label = " with speech normalization" if args.normalize_audio else ""
    log(f"  - Extracting mono 16 kHz audio{normalization_label}")
    extract_audio(
        video_path,
        audio_path,
        normalize_audio=args.normalize_audio,
        normalization_filter=args.normalization_filter,
    )

    transcription_chapters, use_source_chapters = choose_transcription_chapters(
        media_info=media_info,
        ignore_source_chapters=args.ignore_source_chapters,
    )
    initial_prompt = build_initial_prompt(
        media_info.title,
        transcription_chapters,
        args.initial_prompt,
        source_name=media_info.source_path.name,
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
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
    )
    save_transcript(transcript_chapters, output_paths.artifact_dir)

    frames, visual_notes = _extract_visual_context(
        video_path=video_path,
        media_duration=media_info.duration,
        args=args,
        ocr_engine=ocr_engine,
        artifact_dir=output_paths.artifact_dir,
    )

    transcript_info.update(
        {
            "whisper_model": args.whisper_model,
            "whisper_device": args.whisper_device,
            "whisper_compute_type": args.whisper_compute_type,
            "hardware_profile": args.hardware_profile,
            "runtime_reason": args.runtime_reason,
            "local_files_only": args.local_files_only,
            "normalized_audio": args.normalize_audio,
            "normalization_filter": (
                args.normalization_filter if args.normalize_audio else None
            ),
            "sat_model": args.sat_model,
            "used_source_chapters": use_source_chapters,
            "segment_unchaptered": args.segment_unchaptered,
            "use_frames": args.use_frames,
            "use_ocr": args.use_ocr,
            "frame_sample_sec": args.frame_sample_sec,
            "section_seconds": args.section_seconds,
            "selected_frames": len(frames),
            "ocr_notes": len(visual_notes),
        }
    )
    _write_json(output_paths.artifact_dir / "transcript_info.json", transcript_info)

    log("  - Writing ordered multimodal analysis bundle")
    write_analysis_bundle(
        output_paths.artifact_dir / "analysis.json",
        media_info=media_info,
        transcript_chapters=transcript_chapters,
        frames=frames,
        section_seconds=args.section_seconds,
        settings={
            "whisper_model": args.whisper_model,
            "whisper_language": args.whisper_language,
            "normalized_audio": args.normalize_audio,
            "frame_sample_sec": args.frame_sample_sec,
            "frame_min_change": args.frame_min_change,
            "max_frames_per_section": args.max_frames_per_section,
            "ocr_enabled": args.use_ocr and args.use_frames,
        },
        document_path=output_paths.document_path,
    )

    log("  - Segmenting the transcript into readable paragraphs")
    rendered_chapters = build_document_chapters(
        transcript_chapters=transcript_chapters,
        sat_model=sat_model,
        use_source_chapters=use_source_chapters,
        segment_unchaptered=args.segment_unchaptered,
        max_generated_chapter_seconds=args.max_generated_chapter_seconds,
        min_generated_chapter_gap=args.min_generated_chapter_gap,
    )
    rendered_chapters = attach_visual_notes_to_chapters(
        rendered_chapters, visual_notes
    )
    save_rendered_chapters(rendered_chapters, output_paths.artifact_dir)

    markdown = render_markdown(
        title=media_info.title,
        source_path=media_info.source_path,
        chapters=rendered_chapters,
        timestamp_paragraphs=args.timestamp_paragraphs,
        add_table_of_contents=args.add_table_of_contents,
    )
    output_paths.document_path.write_text(markdown, encoding="utf-8")
    log(f"  - Wrote transcript document: {output_paths.document_path}")
    log(f"  - Pi bundle: {output_paths.artifact_dir / 'analysis.json'}")


def _extract_visual_context(
    video_path, media_duration, args, ocr_engine, artifact_dir
) -> tuple[list[FrameObservation], list[VisualNote]]:
    if not args.use_frames:
        _write_json(artifact_dir / "frames.json", [])
        _write_json(artifact_dir / "visual_notes.json", [])
        (artifact_dir / "ocr.jsonl").write_text("", encoding="utf-8")
        return [], []

    ocr_label = " with OCR" if ocr_engine is not None else ""
    log(f"  - Selecting meaningful screen frames{ocr_label}")
    return extract_meaningful_frames(
        video_path=video_path,
        media_duration=media_duration,
        artifact_dir=artifact_dir,
        ocr_engine=ocr_engine,
        sample_sec=args.frame_sample_sec,
        section_sec=args.section_seconds,
        max_frames_per_section=args.max_frames_per_section,
        min_change_score=args.frame_min_change,
        max_dimension=args.frame_max_dimension,
        jpeg_quality=args.frame_jpeg_quality,
        ocr_min_score=args.ocr_min_score,
        ocr_min_chars=args.ocr_min_chars,
        ocr_max_lines=args.ocr_max_lines,
        ocr_max_note_chars=args.ocr_max_note_chars,
        ocr_dedupe_window_sec=args.ocr_dedupe_window_sec,
        verbose_logger=lambda message: vlog(args.verbose, message),
    )


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
