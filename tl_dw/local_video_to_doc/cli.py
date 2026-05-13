from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from tl_dw.common.console import env_flag, log
from tl_dw.common.paths import DEFAULT_VIDEO_DIR

from .fallback_segmenter import SimpleParagraphSegmenter
from .media import resolve_videos
from .ocr import load_ocr_engine
from .pipeline import process_video
from .text_utils import normalize_whitespace
from .transcription import load_whisper_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVP local-video variant of yt2doc.")
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
        help="Output directory, or a .md file path when processing a single video.",
    )
    _add_transcription_args(parser)
    _add_ocr_args(parser)
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=os.getenv("EXTRACTION_DEVICE", "cpu"),
        help="High-level runtime choice for transcription: cpu or gpu.",
    )
    parser.add_argument(
        "--segment-unchaptered",
        action="store_true",
        help="Generate heuristic chapter splits when the source video has no chapters.",
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
        help="Prefix each paragraph with its start timestamp.",
    )
    parser.add_argument(
        "--add-table-of-contents",
        action="store_true",
        help="Add a table of contents when multiple chapters are present.",
    )
    parser.add_argument(
        "--max-generated-chapter-seconds",
        type=float,
        default=float(os.getenv("EXTRACTION_MAX_GENERATED_CHAPTER_SECONDS", "480")),
        help="Approximate max duration for heuristic chapter generation.",
    )
    parser.add_argument(
        "--min-generated-chapter-gap",
        type=float,
        default=float(os.getenv("EXTRACTION_MIN_GENERATED_CHAPTER_GAP", "18")),
        help="Silence gap threshold used when generating heuristic chapters.",
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
    _resolve_runtime_args(args)
    _validate_args(args)

    videos = resolve_videos(args.video, args.video_dir)
    multiple_videos = len(videos) > 1

    log(f"Found {len(videos)} video(s)")
    log(f"Whisper model: {args.whisper_model}")
    log(f"Whisper device: {args.whisper_device}")
    log(f"Whisper compute type: {args.whisper_compute_type}")
    log(f"SaT model: {args.sat_model}")
    log(f"OCR enabled: {args.use_ocr}")

    whisper_model = load_whisper_model(
        model_name=args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute_type,
    )
    sat_model = _load_sat_model(args.sat_model)
    ocr_engine = load_ocr_engine() if args.use_ocr else None

    for video_path in videos:
        process_video(
            video_path=video_path,
            args=args,
            whisper_model=whisper_model,
            sat_model=sat_model,
            ocr_engine=ocr_engine,
            multiple_videos=multiple_videos,
        )

    log("\nAll done.")


def _load_sat_model(model_name: str):
    try:
        from wtpsplit import SaT
    except ImportError:  # pragma: no cover
        log("SaT is unavailable, falling back to the built-in paragraph segmenter.")
        return SimpleParagraphSegmenter()

    try:
        return SaT(model_name)
    except Exception as exc:  # pragma: no cover
        log(
            f"SaT model load failed, falling back to the built-in paragraph segmenter: {exc}"
        )
        return SimpleParagraphSegmenter()


def _add_transcription_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--whisper-model",
        default=os.getenv("EXTRACTION_WHISPER_MODEL", "small"),
        help="faster-whisper model name or local model path.",
    )
    parser.add_argument(
        "--whisper-device",
        default=os.getenv("EXTRACTION_WHISPER_DEVICE"),
        help="Whisper device, for example cpu or cuda.",
    )
    parser.add_argument(
        "--whisper-compute-type",
        default=os.getenv("EXTRACTION_WHISPER_COMPUTE_TYPE"),
        help="Whisper compute type, for example int8, float16, or int8_float16.",
    )
    parser.add_argument(
        "--whisper-language",
        default=os.getenv("EXTRACTION_WHISPER_LANGUAGE", ""),
        help="Optional language code such as it or en.",
    )
    parser.add_argument(
        "--initial-prompt",
        default=os.getenv("EXTRACTION_INITIAL_PROMPT", ""),
        help="Optional transcription prompt override.",
    )
    parser.add_argument(
        "--sat-model",
        default=os.getenv("EXTRACTION_SAT_MODEL", "sat-3l-sm"),
        help="wtpsplit SaT model used for paragraph segmentation.",
    )


def _add_ocr_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-ocr",
        dest="use_ocr",
        action="store_false",
        default=not env_flag("EXTRACTION_DISABLE_OCR", False),
        help="Disable OCR-based visual context extraction.",
    )
    parser.add_argument(
        "--ocr-sample-sec",
        type=float,
        default=float(os.getenv("EXTRACTION_OCR_SAMPLE_SEC", "8.0")),
        help="How often to sample video frames for OCR.",
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
        help="Minimum text length for general OCR lines unless a meeting/UI hint is present.",
    )
    parser.add_argument(
        "--ocr-max-lines",
        type=int,
        default=int(os.getenv("EXTRACTION_OCR_MAX_LINES", "6")),
        help="Maximum OCR lines to keep per visual note.",
    )
    parser.add_argument(
        "--ocr-max-note-chars",
        type=int,
        default=int(os.getenv("EXTRACTION_OCR_MAX_NOTE_CHARS", "240")),
        help="Maximum rendered length of a visual note.",
    )
    parser.add_argument(
        "--ocr-dedupe-window-sec",
        type=float,
        default=float(os.getenv("EXTRACTION_OCR_DEDUPE_WINDOW_SEC", "45.0")),
        help="Time window used to suppress repeated OCR notes from nearby frames.",
    )


def _validate_args(args: argparse.Namespace) -> None:
    checks = [
        (args.ocr_sample_sec > 0, "--ocr-sample-sec must be greater than 0."),
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
    requested_device = (args.device or "cpu").strip().lower()
    if requested_device not in {"cpu", "gpu"}:
        raise SystemExit("--device must be either cpu or gpu.")

    if args.whisper_device:
        args.whisper_device = args.whisper_device.strip()
    else:
        args.whisper_device = "cuda" if requested_device == "gpu" else "cpu"

    if args.whisper_compute_type:
        args.whisper_compute_type = args.whisper_compute_type.strip()
    else:
        args.whisper_compute_type = (
            "int8_float16" if requested_device == "gpu" else "int8"
        )
