from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from tl_dw.common.console import env_flag, log
from tl_dw.common.paths import DEFAULT_VIDEO_DIR

from .fallback_segmenter import SimpleParagraphSegmenter
from .media import DEFAULT_NORMALIZATION_FILTER, resolve_videos
from .ocr import load_ocr_engine
from .pipeline import process_video
from .text_utils import normalize_whitespace
from .transcription import (
    HARDWARE_PROFILE_CHOICES,
    load_whisper_model,
    resolve_runtime_settings,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Turn meeting recordings into a timestamped transcript, selected "
            "screen frames, and a Pi-ready multimodal analysis bundle."
        )
    )
    parser.add_argument(
        "--video", type=Path, help="Single local video file to process."
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Directory to scan when --video is omitted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory, or a .md file path when processing one video.",
    )
    _add_transcription_args(parser)
    _add_frame_args(parser)
    parser.add_argument(
        "--segment-unchaptered",
        action="store_true",
        help="Generate heuristic document chapters when the source has none.",
    )
    parser.add_argument(
        "--ignore-source-chapters",
        "--ignore-chapters",
        dest="ignore_source_chapters",
        action="store_true",
        help="Ignore embedded source chapters and treat the video as unchaptered.",
    )
    parser.add_argument(
        "--timestamp-paragraphs",
        action="store_true",
        help="Prefix each rendered transcript paragraph with its start timestamp.",
    )
    parser.add_argument(
        "--add-table-of-contents",
        action="store_true",
        help="Add a table of contents when multiple document chapters exist.",
    )
    parser.add_argument(
        "--max-generated-chapter-seconds",
        type=float,
        default=float(os.getenv("EXTRACTION_MAX_GENERATED_CHAPTER_SECONDS", "480")),
        help="Approximate max duration for heuristic document chapters.",
    )
    parser.add_argument(
        "--min-generated-chapter-gap",
        type=float,
        default=float(os.getenv("EXTRACTION_MIN_GENERATED_CHAPTER_GAP", "18")),
        help="Speech-gap threshold used when generating document chapters.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=env_flag("EXTRACTION_VERBOSE", False),
        help="Enable verbose progress logging.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.whisper_language = normalize_whitespace(args.whisper_language) or None
    args.initial_prompt = normalize_whitespace(args.initial_prompt) or None
    args.normalization_filter = args.normalization_filter.strip()
    _resolve_runtime_args(args)
    _validate_args(args)

    videos = resolve_videos(args.video, args.video_dir)
    multiple_videos = len(videos) > 1

    log(f"Found {len(videos)} video(s)")
    log(f"Whisper model: {args.whisper_model}")
    log(f"Whisper device: {args.whisper_device}")
    log(f"Whisper compute type: {args.whisper_compute_type}")
    log(f"Runtime selection: {args.runtime_reason}")
    log(f"Audio normalization: {args.normalize_audio}")
    log(f"Visual frames: {args.use_frames}; OCR: {args.use_ocr}")

    whisper_model = load_whisper_model(
        model_name=args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute_type,
        local_files_only=args.local_files_only,
    )
    sat_model = _load_sat_model(args.sat_model)
    ocr_engine = (
        load_ocr_engine() if args.use_frames and args.use_ocr else None
    )

    try:
        for video_path in videos:
            process_video(
                video_path=video_path,
                args=args,
                whisper_model=whisper_model,
                sat_model=sat_model,
                ocr_engine=ocr_engine,
                multiple_videos=multiple_videos,
            )
    finally:
        if ocr_engine is not None and hasattr(ocr_engine, "close"):
            ocr_engine.close()

    log("\nAll done.")


def _load_sat_model(model_name: str):
    if model_name.strip().lower() in {"simple", "builtin", "built-in"}:
        return SimpleParagraphSegmenter()

    try:
        from wtpsplit import SaT
    except ImportError:  # pragma: no cover
        log("SaT is unavailable, falling back to the built-in paragraph segmenter.")
        return SimpleParagraphSegmenter()

    try:
        return SaT(model_name)
    except Exception as exc:  # pragma: no cover - model/network/runtime dependent
        log(
            f"SaT model load failed, falling back to the built-in paragraph segmenter: {exc}"
        )
        return SimpleParagraphSegmenter()


def _add_transcription_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--whisper-model",
        default=os.getenv("EXTRACTION_WHISPER_MODEL", "large-v3"),
        help="faster-whisper model name or local model path (default: large-v3).",
    )
    parser.add_argument(
        "--hardware-profile",
        choices=HARDWARE_PROFILE_CHOICES,
        default=os.getenv("EXTRACTION_HARDWARE_PROFILE", "auto"),
        help=(
            "Runtime preset: auto, CPU/MacBook INT8, Pascal CUDA INT8, or modern "
            "CUDA FP16 (default: auto)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=os.getenv("EXTRACTION_DEVICE") or None,
        help="Compatibility shortcut overriding the profile with cpu or gpu.",
    )
    parser.add_argument(
        "--whisper-device",
        default=os.getenv("EXTRACTION_WHISPER_DEVICE"),
        help="Low-level Whisper device override, for example cpu or cuda.",
    )
    parser.add_argument(
        "--whisper-compute-type",
        default=os.getenv("EXTRACTION_WHISPER_COMPUTE_TYPE"),
        help="Low-level CTranslate2 compute type override.",
    )
    parser.add_argument(
        "--whisper-language",
        default=os.getenv("EXTRACTION_WHISPER_LANGUAGE", ""),
        help="Optional language code such as it or en; empty enables detection.",
    )
    parser.add_argument(
        "--initial-prompt",
        default=os.getenv("EXTRACTION_INITIAL_PROMPT", ""),
        help="Optional vocabulary or transcription-style hint.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=int(os.getenv("EXTRACTION_BEAM_SIZE", "5")),
        help="Whisper decoding beam size (default: 5).",
    )
    parser.add_argument(
        "--no-vad",
        dest="vad_filter",
        action="store_false",
        default=True,
        help="Disable Whisper voice-activity filtering.",
    )
    parser.add_argument(
        "--normalize-audio",
        action=argparse.BooleanOptionalAction,
        default=env_flag("EXTRACTION_NORMALIZE_AUDIO", True),
        help="Apply speech-focused FFmpeg normalization (default: enabled).",
    )
    parser.add_argument(
        "--normalization-filter",
        default=os.getenv(
            "EXTRACTION_NORMALIZATION_FILTER", DEFAULT_NORMALIZATION_FILTER
        ),
        help="FFmpeg audio filter used when normalization is enabled.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load an already cached Hugging Face model or local model path.",
    )
    parser.add_argument(
        "--sat-model",
        default=os.getenv("EXTRACTION_SAT_MODEL", "simple"),
        help=(
            "Paragraph segmenter: simple (built in), or a wtpsplit SaT model "
            "when installed with `uv sync --extra sat`."
        ),
    )


def _add_frame_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-frames",
        dest="use_frames",
        action="store_false",
        default=not env_flag("EXTRACTION_DISABLE_FRAMES", False),
        help="Disable meaningful-frame extraction entirely.",
    )
    parser.add_argument(
        "--no-ocr",
        dest="use_ocr",
        action="store_false",
        default=not env_flag("EXTRACTION_DISABLE_OCR", False),
        help="Keep selected images but disable local OCR enrichment.",
    )
    parser.add_argument(
        "--frame-sample-sec",
        "--ocr-sample-sec",
        dest="frame_sample_sec",
        type=float,
        default=float(os.getenv("EXTRACTION_FRAME_SAMPLE_SEC", "5.0")),
        help="Interval for inexpensive visual scanning (default: 5 seconds).",
    )
    parser.add_argument(
        "--section-seconds",
        type=float,
        default=float(os.getenv("EXTRACTION_SECTION_SECONDS", "120")),
        help="Target duration of each Pi analysis section (default: 120 seconds).",
    )
    parser.add_argument(
        "--max-frames-per-section",
        type=int,
        default=int(os.getenv("EXTRACTION_MAX_FRAMES_PER_SECTION", "4")),
        help="Maximum retained images in each analysis section (default: 4).",
    )
    parser.add_argument(
        "--frame-min-change",
        type=float,
        default=float(os.getenv("EXTRACTION_FRAME_MIN_CHANGE", "0.015")),
        help="Minimum normalized novelty score for non-anchor frames.",
    )
    parser.add_argument(
        "--frame-max-dimension",
        type=int,
        default=int(os.getenv("EXTRACTION_FRAME_MAX_DIMENSION", "1600")),
        help="Maximum retained frame width in pixels (default: 1600).",
    )
    parser.add_argument(
        "--frame-jpeg-quality",
        type=int,
        default=int(os.getenv("EXTRACTION_FRAME_JPEG_QUALITY", "3")),
        help="FFmpeg JPEG qscale (2 is highest; default: 3).",
    )
    parser.add_argument(
        "--ocr-min-score",
        type=float,
        default=float(os.getenv("EXTRACTION_OCR_MIN_SCORE", "0.55")),
        help="Minimum OCR confidence for keeping a line.",
    )
    parser.add_argument(
        "--ocr-min-chars",
        type=int,
        default=int(os.getenv("EXTRACTION_OCR_MIN_CHARS", "6")),
        help="Minimum generic OCR line length.",
    )
    parser.add_argument(
        "--ocr-max-lines",
        type=int,
        default=int(os.getenv("EXTRACTION_OCR_MAX_LINES", "12")),
        help="Maximum OCR lines retained per selected frame.",
    )
    parser.add_argument(
        "--ocr-max-note-chars",
        type=int,
        default=int(os.getenv("EXTRACTION_OCR_MAX_NOTE_CHARS", "800")),
        help="Maximum rendered OCR context length per frame.",
    )
    parser.add_argument(
        "--ocr-dedupe-window-sec",
        type=float,
        default=float(os.getenv("EXTRACTION_OCR_DEDUPE_WINDOW_SEC", "45")),
        help="Window for suppressing repeated OCR notes in the transcript Markdown.",
    )


def _validate_args(args: argparse.Namespace) -> None:
    checks = [
        (args.beam_size >= 1, "--beam-size must be at least 1."),
        (bool(args.normalization_filter), "--normalization-filter cannot be empty."),
        (args.frame_sample_sec > 0, "--frame-sample-sec must be greater than 0."),
        (args.section_seconds > 0, "--section-seconds must be greater than 0."),
        (
            args.max_frames_per_section > 0,
            "--max-frames-per-section must be greater than 0.",
        ),
        (
            0 <= args.frame_min_change <= 1,
            "--frame-min-change must be between 0 and 1.",
        ),
        (
            args.frame_max_dimension >= 320,
            "--frame-max-dimension must be at least 320.",
        ),
        (
            2 <= args.frame_jpeg_quality <= 31,
            "--frame-jpeg-quality must be between 2 and 31.",
        ),
        (0 <= args.ocr_min_score <= 1, "--ocr-min-score must be between 0 and 1."),
        (args.ocr_min_chars > 0, "--ocr-min-chars must be greater than 0."),
        (args.ocr_max_lines > 0, "--ocr-max-lines must be greater than 0."),
        (args.ocr_max_note_chars > 0, "--ocr-max-note-chars must be greater than 0."),
        (
            args.ocr_dedupe_window_sec >= 0,
            "--ocr-dedupe-window-sec must be 0 or greater.",
        ),
        (
            args.max_generated_chapter_seconds > 0,
            "--max-generated-chapter-seconds must be greater than 0.",
        ),
        (
            args.min_generated_chapter_gap >= 0,
            "--min-generated-chapter-gap must be 0 or greater.",
        ),
    ]
    for passed, message in checks:
        if not passed:
            raise SystemExit(message)


def _resolve_runtime_args(args: argparse.Namespace) -> None:
    profile = getattr(args, "hardware_profile", "auto") or "auto"
    settings = resolve_runtime_settings(profile)
    device = settings.device
    compute_type = settings.compute_type
    reason = settings.reason

    requested_device = getattr(args, "device", None)
    if requested_device:
        if requested_device not in {"cpu", "gpu"}:
            raise SystemExit("--device must be either cpu or gpu.")
        device = "cuda" if requested_device == "gpu" else "cpu"
        compute_type = "int8_float16" if requested_device == "gpu" else "int8"
        reason += f"; --device overrode the profile with {requested_device}"

    if getattr(args, "whisper_device", None):
        device = args.whisper_device.strip()
        if not device:
            raise SystemExit("--whisper-device cannot be empty.")
        reason += f"; low-level device override selected {device}"

    if getattr(args, "whisper_compute_type", None):
        compute_type = args.whisper_compute_type.strip()
        if not compute_type:
            raise SystemExit("--whisper-compute-type cannot be empty.")
        reason += f"; low-level compute override selected {compute_type}"

    args.whisper_device = device
    args.whisper_compute_type = compute_type
    args.runtime_reason = reason
