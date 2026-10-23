from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tl_dw.common.console import log, vlog

from .analysis_bundle import write_analysis_bundle
from .artifacts import (
    PreparationCache,
    artifact_file,
    atomic_json,
    preparation_lock,
    sha256_file,
    signature,
)
from .chapters import attach_visual_notes_to_chapters, build_document_chapters
from .documents import prepare_documents
from .frames import extract_meaningful_frames
from .media import (
    build_initial_prompt,
    choose_transcription_chapters,
    extract_audio,
    ffprobe_metadata,
    media_info_from_metadata,
    resolve_output_paths,
    summary_markdown_path,
)
from .models import FrameObservation, TranscriptChapter, TranscriptSegment, VisualNote
from .preflight import (
    media_preflight,
    require_audio_selection,
    require_media_tools,
    select_audio,
)
from .rendering import render_markdown, save_rendered_chapters
from .text_utils import build_summary_title
from .transcription import save_transcript, transcribe_media

PREPARATION_VERSION = 1


def process_video(
    video_path,
    args: Namespace,
    whisper_model: Any,
    sat_model: Any,
    ocr_engine: Any | None,
    multiple_videos: bool,
    resource_loader=None,
) -> None:
    require_media_tools()
    video_path = Path(video_path).resolve()
    log(f"\n=== Processing: {video_path.name} ===")
    paths = resolve_output_paths(video_path, args.output, multiple_videos)
    root = paths.artifact_dir.resolve()
    # Refuse pre-existing links for all fixed outputs before a stage can overwrite them.
    for name in (
        "metadata.json",
        "preflight.json",
        "audio-selection.json",
        "audio.wav",
        ".audio.partial.wav",
        "transcript.txt",
        "transcript.jsonl",
        "asr.json",
        "frames",
        "frames.json",
        "ocr.jsonl",
        "visual_notes.json",
        "transcript_info.json",
        "chapters.json",
        "analysis.json",
        "provenance.json",
    ):
        target = root / name
        if target.is_symlink():
            raise ValueError(f"Refusing symlink in preparation outputs: {target}")
    with preparation_lock(root):
        _prepare(
            video_path,
            args,
            paths,
            whisper_model,
            sat_model,
            ocr_engine,
            resource_loader,
        )


def _prepare(video, args, paths, whisper_model, sat_model, ocr_engine, resource_loader):
    root = paths.artifact_dir.resolve()

    def resource(name, provided):
        return resource_loader(name) if resource_loader else provided

    source_stat = video.stat()
    metadata = ffprobe_metadata(video)
    media_info = media_info_from_metadata(video, metadata)
    atomic_json(root / "metadata.json", metadata)
    preflight = media_preflight(video, metadata, root, args)
    log(
        f"  - Preflight: {media_info.duration:.1f}s, {video.stat().st_size} bytes, "
        f"{len(preflight['audio'])} audio track(s); estimated temporary {preflight['estimated_temporary_bytes']} bytes"
    )
    if getattr(args, "preflight_only", False):
        log(f"  - Preflight report: {root / 'preflight.json'}")
        return
    if (
        (root / "analysis.json").exists()
        and not (root / "preparation.json").exists()
        and not getattr(args, "force_preparation", False)
    ):
        raise ValueError(
            "Existing legacy analysis.json has no verified preparation provenance. Reuse it with tl_dw_analyze or /tl-dw-analyze; use --force-preparation only to explicitly rebuild ASR/frames."
        )
    source = {
        "path": str(video),
        "sha256": sha256_file(video),
        "bytes": source_stat.st_size,
    }
    if (
        video.stat().st_size != source_stat.st_size
        or video.stat().st_mtime_ns != source_stat.st_mtime_ns
    ):
        raise ValueError("Recording changed during probing/hashing; use a stable input")
    cache = PreparationCache(root, force=getattr(args, "force_preparation", False))
    job_key = signature([PREPARATION_VERSION, source])
    cache.begin("preparation", job_key)
    selection_key = signature(
        [
            PREPARATION_VERSION,
            source,
            getattr(args, "audio_stream", None),
            getattr(args, "audio_mix", None),
            getattr(args, "audio_diagnostics", "full"),
        ]
    )
    if cache.reusable("audio-selection", selection_key):
        selection = _read(root / "audio-selection.json")
        log("  - Reusing verified audio selection")
    else:
        cache.begin("audio-selection", selection_key)
        selection = select_audio(
            video,
            metadata,
            stream=getattr(args, "audio_stream", None),
            mix=getattr(args, "audio_mix", None),
            scope=getattr(args, "audio_diagnostics", "full"),
        )
        atomic_json(root / "audio-selection.json", selection)
        cache.complete(
            "audio-selection", selection_key, [root / "audio-selection.json"]
        )
    selected = require_audio_selection(selection)
    audio_key = signature(
        [
            PREPARATION_VERSION,
            source,
            selected,
            args.normalize_audio,
            args.normalization_filter,
        ]
    )
    audio = root / "audio.wav"
    if not cache.reusable("audio", audio_key):
        cache.begin("audio", audio_key)
        log(f"  - Extracting mono 16 kHz audio from stream(s) {selected}")
        temporary = root / ".audio.partial.wav"
        try:
            extract_audio(
                video,
                temporary,
                normalize_audio=args.normalize_audio,
                normalization_filter=args.normalization_filter,
                audio_streams=selected,
            )
            temporary.replace(audio)
        finally:
            temporary.unlink(missing_ok=True)
        cache.complete("audio", audio_key, [audio])
    else:
        log("  - Reusing verified extracted audio")

    chapters, use_source_chapters = choose_transcription_chapters(
        media_info, args.ignore_source_chapters
    )
    prompt = build_initial_prompt(
        media_info.title, chapters, args.initial_prompt, source_name=video.name
    )
    asr_key = signature(
        [
            PREPARATION_VERSION,
            audio_key,
            sha256_file(audio),
            [asdict(c) for c in chapters],
            prompt,
            args.whisper_model,
            args.whisper_language,
            args.whisper_device,
            args.whisper_compute_type,
            args.beam_size,
            args.vad_filter,
        ]
    )
    if cache.reusable("transcription", asr_key):
        saved = _read(root / "asr.json")
        transcript = [
            TranscriptChapter(
                title=c["title"],
                segments=[TranscriptSegment(**s) for s in c["segments"]],
            )
            for c in saved["chapters"]
        ]
        info = saved["info"]
        log("  - Reusing verified speech transcription")
    else:
        cache.begin("transcription", asr_key)
        log("  - Transcribing speech")
        transcript, info = transcribe_media(
            audio_path=audio,
            media_duration=media_info.duration,
            chapters=chapters,
            whisper_model=resource("whisper", whisper_model),
            requested_language=args.whisper_language,
            initial_prompt=prompt,
            verbose_logger=lambda message: vlog(args.verbose, message),
            beam_size=args.beam_size,
            vad_filter=args.vad_filter,
        )
        save_transcript(transcript, root)
        atomic_json(
            root / "asr.json",
            {"chapters": [asdict(c) for c in transcript], "info": info},
        )
        cache.complete(
            "transcription",
            asr_key,
            [root / "asr.json", root / "transcript.txt", root / "transcript.jsonl"],
        )

    visual_settings = {
        name: getattr(args, name)
        for name in (
            "use_frames",
            "use_ocr",
            "frame_sample_sec",
            "section_seconds",
            "max_frames_per_section",
            "frame_min_change",
            "frame_max_dimension",
            "frame_jpeg_quality",
            "ocr_min_score",
            "ocr_min_chars",
            "ocr_max_lines",
            "ocr_max_note_chars",
            "ocr_dedupe_window_sec",
        )
    }
    visual_key = signature([PREPARATION_VERSION, source, visual_settings])
    if cache.reusable("visuals", visual_key):
        frames = [FrameObservation(**f) for f in _read(root / "frames.json")]
        notes = [VisualNote(**n) for n in _read(root / "visual_notes.json")]
        log("  - Reusing verified selected frames and OCR")
    else:
        cache.begin("visuals", visual_key)
        frames, notes = _extract_visual_context(
            video,
            media_info.duration,
            args,
            resource("ocr", ocr_engine) if args.use_frames and args.use_ocr else None,
            root,
        )
        cache.complete(
            "visuals",
            visual_key,
            [
                root / "frames.json",
                root / "visual_notes.json",
                root / "ocr.jsonl",
                *(artifact_file(root, f.image_path) for f in frames),
            ],
        )

    render_key = signature(
        [
            PREPARATION_VERSION,
            asr_key,
            visual_key,
            cache.stages["transcription"]["outputs"],
            cache.stages["visuals"]["outputs"],
            args.sat_model,
            args.segment_unchaptered,
            args.max_generated_chapter_seconds,
            args.min_generated_chapter_gap,
            args.timestamp_paragraphs,
            args.add_table_of_contents,
            str(paths.document_path),
        ]
    )
    external_document = not paths.document_path.resolve().is_relative_to(root)
    if external_document or not cache.reusable("bundle", render_key):
        cache.begin("bundle", render_key)
        info = {
            **info,
            "whisper_model": args.whisper_model,
            "whisper_device": args.whisper_device,
            "whisper_compute_type": args.whisper_compute_type,
            "hardware_profile": args.hardware_profile,
            "runtime_reason": args.runtime_reason,
            "local_files_only": args.local_files_only,
            "normalized_audio": args.normalize_audio,
            "normalization_filter": args.normalization_filter
            if args.normalize_audio
            else None,
            "sat_model": args.sat_model,
            "used_source_chapters": use_source_chapters,
            "segment_unchaptered": args.segment_unchaptered,
            **visual_settings,
            "selected_frames": len(frames),
            "ocr_notes": len(notes),
            "selected_audio_streams": selected,
        }
        atomic_json(root / "transcript_info.json", info)
        rendered = build_document_chapters(
            transcript_chapters=transcript,
            sat_model=resource("sat", sat_model),
            use_source_chapters=use_source_chapters,
            segment_unchaptered=args.segment_unchaptered,
            max_generated_chapter_seconds=args.max_generated_chapter_seconds,
            min_generated_chapter_gap=args.min_generated_chapter_gap,
        )
        rendered = attach_visual_notes_to_chapters(rendered, notes)
        save_rendered_chapters(rendered, root)
        title = build_summary_title(
            [p.text for c in rendered for p in c.paragraphs], fallback=media_info.title
        )
        document = summary_markdown_path(paths.document_path, paths.artifact_dir, title)
        if document.is_symlink():
            raise ValueError("Refusing a symlink transcript document")
        document.write_text(
            render_markdown(
                title=title,
                source_path=video,
                chapters=rendered,
                timestamp_paragraphs=args.timestamp_paragraphs,
                add_table_of_contents=args.add_table_of_contents,
            ),
            encoding="utf-8",
        )
        write_analysis_bundle(
            root / "analysis.json",
            media_info=media_info,
            transcript_chapters=transcript,
            frames=frames,
            section_seconds=args.section_seconds,
            settings={
                "whisper_model": args.whisper_model,
                "whisper_language": args.whisper_language,
                "normalized_audio": args.normalize_audio,
                **visual_settings,
            },
            document_path=document,
            title=title,
        )
        # An explicitly external .md output is not a reusable artifact: do not hash
        # files outside this root or make the cache an arbitrary file reader.
        render_outputs = [
            root / "analysis.json",
            root / "chapters.json",
            root / "transcript_info.json",
        ]
        if document.resolve().is_relative_to(root):
            render_outputs.append(document)
        cache.complete("bundle", render_key, render_outputs)
        log(f"  - Wrote transcript document: {document}")
    else:
        log("  - Reusing verified analysis bundle")
    documents = None
    if getattr(args, "attachment", None):
        # Share the journal instance: independent stale snapshots would lose the
        # document stage when this job completes.
        documents = prepare_documents(args.attachment, root, cache=cache)
    if (
        video.stat().st_size != source_stat.st_size
        or video.stat().st_mtime_ns != source_stat.st_mtime_ns
    ):
        raise ValueError(
            "Recording changed during preparation; retry with a stable input"
        )
    cache.complete("preparation", job_key, [root / "analysis.json"])
    atomic_json(
        root / "provenance.json",
        {
            "schema_version": 1,
            "preparation_version": PREPARATION_VERSION,
            "source": source,
            "audio_selection": selection,
            "stages": _read(root / "preparation.json")["stages"],
            "documents_signature": documents["signature"] if documents else None,
            "compression_applied": False,
            "this_invocation": {
                "executed_stages": sorted(cache.executed - {"preparation"}),
                "reused_stages": sorted(cache.reused),
                "source_probed_and_hashed": True,
            },
        },
    )
    log(f"  - Pi bundle: {root / 'analysis.json'}")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_visual_context(
    video_path, media_duration, args, ocr_engine, artifact_dir
) -> tuple[list[FrameObservation], list[VisualNote]]:
    if not args.use_frames:
        atomic_json(artifact_dir / "frames.json", [])
        atomic_json(artifact_dir / "visual_notes.json", [])
        (artifact_dir / "ocr.jsonl").write_text("", encoding="utf-8")
        return [], []
    log(
        "  - Selecting meaningful screen frames"
        + (" with OCR" if ocr_engine is not None else "")
    )
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
