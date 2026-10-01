from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageChops, ImageFilter, ImageStat

from tl_dw.common.console import log
from tl_dw.common.process import run_command

from .models import FrameObservation, VisualNote
from .ocr_filters import (
    dedupe_visual_notes,
    filter_meaningful_ocr_lines,
    normalize_ocr_text,
    summarize_visual_lines,
)


@dataclass(frozen=True)
class SampledFrame:
    timestamp: float
    path: Path
    change_score: float


def extract_meaningful_frames(
    video_path: Path,
    media_duration: float,
    artifact_dir: Path,
    *,
    ocr_engine: Any | None,
    sample_sec: float,
    section_sec: float,
    max_frames_per_section: int,
    min_change_score: float,
    max_dimension: int,
    jpeg_quality: int,
    ocr_min_score: float,
    ocr_min_chars: int,
    ocr_max_lines: int,
    ocr_max_note_chars: int,
    ocr_dedupe_window_sec: float,
    verbose_logger: callable,
) -> tuple[list[FrameObservation], list[VisualNote]]:
    """Select representative screen states and optionally enrich them with OCR.

    A low-rate FFmpeg scan is substantially cheaper than decoding random frames one at
    a time. Selection keeps an anchor for every analysis section, then adds the most
    visually novel states while enforcing temporal diversity.
    """
    if sample_sec <= 0:
        raise ValueError("sample_sec must be greater than 0")
    if section_sec <= 0:
        raise ValueError("section_sec must be greater than 0")
    if max_frames_per_section <= 0:
        raise ValueError("max_frames_per_section must be greater than 0")

    frames_dir = artifact_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tl-dw-frame-scan-") as temp_dir_name:
        scan_dir = Path(temp_dir_name)
        _extract_scan_frames(
            video_path=video_path,
            output_dir=scan_dir,
            sample_sec=sample_sec,
            max_dimension=max_dimension,
            jpeg_quality=jpeg_quality,
        )
        sampled = score_sampled_frames(scan_dir, sample_sec=sample_sec)
        selected = select_sampled_frames(
            sampled,
            media_duration=media_duration,
            section_sec=section_sec,
            max_frames_per_section=max_frames_per_section,
            min_change_score=min_change_score,
            min_spacing_sec=sample_sec,
        )

        observations, notes, ocr_rows = _materialize_selected_frames(
            selected=selected,
            frames_dir=frames_dir,
            artifact_dir=artifact_dir,
            ocr_engine=ocr_engine,
            ocr_min_score=ocr_min_score,
            ocr_min_chars=ocr_min_chars,
            ocr_max_lines=ocr_max_lines,
            ocr_max_note_chars=ocr_max_note_chars,
            verbose_logger=verbose_logger,
        )

    ocr_failure_count = sum(bool(row.get("error")) for row in ocr_rows)
    if ocr_failure_count:
        log(
            f"  ! OCR failed for {ocr_failure_count}/{len(observations)} selected "
            "frames; the images were kept for vision analysis."
        )
    notes = dedupe_visual_notes(
        notes, dedupe_window_sec=ocr_dedupe_window_sec
    )
    _write_frame_observations(artifact_dir / "frames.json", observations)
    _write_visual_notes(artifact_dir / "visual_notes.json", notes)
    _write_jsonl(artifact_dir / "ocr.jsonl", ocr_rows)
    return observations, notes


def _extract_scan_frames(
    video_path: Path,
    output_dir: Path,
    sample_sec: float,
    max_dimension: int,
    jpeg_quality: int,
) -> None:
    output_pattern = output_dir / "%06d.jpg"
    video_filter = (
        f"fps=1/{sample_sec:.6f}:start_time=0,"
        f"scale=w='min(iw,{max_dimension})':h=-2"
    )
    run_command(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-an",
            "-vf",
            video_filter,
            "-q:v",
            str(jpeg_quality),
            str(output_pattern),
        ]
    )
    if not any(output_dir.glob("*.jpg")):
        raise RuntimeError(f"FFmpeg did not extract any frames from {video_path}")


def score_sampled_frames(scan_dir: Path, sample_sec: float) -> list[SampledFrame]:
    paths = sorted(scan_dir.glob("*.jpg"))
    sampled: list[SampledFrame] = []
    previous: Image.Image | None = None

    for index, path in enumerate(paths):
        with Image.open(path) as source:
            current = source.convert("L").resize((256, 144), Image.Resampling.BILINEAR)
        score = 0.0 if previous is None else frame_change_score(previous, current)
        sampled.append(
            SampledFrame(
                timestamp=round(index * sample_sec, 3),
                path=path,
                change_score=score,
            )
        )
        previous = current

    return sampled


def frame_change_score(previous: Image.Image, current: Image.Image) -> float:
    """Return a 0..1-ish screen-change score using pixels and edge structure."""
    pixel_delta = ImageStat.Stat(ImageChops.difference(previous, current)).mean[0] / 255
    previous_edges = previous.filter(ImageFilter.FIND_EDGES)
    current_edges = current.filter(ImageFilter.FIND_EDGES)
    edge_delta = (
        ImageStat.Stat(ImageChops.difference(previous_edges, current_edges)).mean[0]
        / 255
    )
    return round((0.65 * pixel_delta) + (0.35 * edge_delta), 6)


def select_sampled_frames(
    sampled: Sequence[SampledFrame],
    *,
    media_duration: float,
    section_sec: float,
    max_frames_per_section: int,
    min_change_score: float,
    min_spacing_sec: float,
) -> list[tuple[SampledFrame, list[str]]]:
    if not sampled:
        return []

    effective_duration = max(
        media_duration,
        sampled[-1].timestamp + 0.001,
    )
    section_count = max(1, math.ceil(effective_duration / section_sec))
    selected: list[tuple[SampledFrame, list[str]]] = []

    for section_index in range(section_count):
        start = section_index * section_sec
        end = min(effective_duration, start + section_sec)
        section_frames = [
            frame
            for frame in sampled
            if start <= frame.timestamp < end
            or (
                section_index == section_count - 1
                and math.isclose(frame.timestamp, end, abs_tol=0.001)
            )
        ]
        if not section_frames:
            continue

        anchor = section_frames[0]
        section_selected = [anchor]
        candidates = [
            frame
            for frame in section_frames[1:]
            if frame.change_score >= min_change_score
        ]

        while candidates and len(section_selected) < max_frames_per_section:
            spaced = [
                candidate
                for candidate in candidates
                if all(
                    abs(candidate.timestamp - chosen.timestamp) >= min_spacing_sec
                    for chosen in section_selected
                )
            ]
            if not spaced:
                break

            def utility(candidate: SampledFrame) -> float:
                nearest_distance = min(
                    abs(candidate.timestamp - chosen.timestamp)
                    for chosen in section_selected
                )
                coverage_bonus = 0.02 * min(1.0, nearest_distance / section_sec)
                return candidate.change_score + coverage_bonus

            chosen = max(spaced, key=lambda frame: (utility(frame), -frame.timestamp))
            section_selected.append(chosen)
            candidates.remove(chosen)

        for frame in sorted(section_selected, key=lambda item: item.timestamp):
            reasons: list[str] = []
            if frame is anchor:
                reasons.append("section-anchor")
            if frame.change_score >= 0.18:
                reasons.append("scene-change")
            elif frame.change_score >= min_change_score:
                reasons.append("visual-change")
            selected.append((frame, reasons))

    return selected


def _materialize_selected_frames(
    *,
    selected: Sequence[tuple[SampledFrame, list[str]]],
    frames_dir: Path,
    artifact_dir: Path,
    ocr_engine: Any | None,
    ocr_min_score: float,
    ocr_min_chars: int,
    ocr_max_lines: int,
    ocr_max_note_chars: int,
    verbose_logger: callable,
) -> tuple[list[FrameObservation], list[VisualNote], list[dict[str, Any]]]:
    observations: list[FrameObservation] = []
    notes: list[VisualNote] = []
    ocr_rows: list[dict[str, Any]] = []

    for index, (sample, reasons) in enumerate(selected, start=1):
        timestamp_label = f"{sample.timestamp:010.3f}".replace(".", "-")
        destination = frames_dir / f"frame-{index:04d}-{timestamp_label}.jpg"
        shutil.copy2(sample.path, destination)
        relative_path = destination.relative_to(artifact_dir).as_posix()

        raw_lines: list[tuple[str, float]] = []
        kept_lines: list[tuple[str, float]] = []
        note_text = ""
        confidence: float | None = None
        ocr_error: str | None = None
        if ocr_engine is not None:
            try:
                result, _ = ocr_engine(str(destination))
                raw_lines = extract_raw_ocr_lines(result or [])
                kept_lines = filter_meaningful_ocr_lines(
                    raw_lines,
                    min_score=ocr_min_score,
                    min_chars=ocr_min_chars,
                )[:ocr_max_lines]
                note_text = summarize_visual_lines(
                    [text for text, _ in kept_lines],
                    max_lines=ocr_max_lines,
                    max_chars=ocr_max_note_chars,
                )
                if kept_lines:
                    confidence = sum(score for _, score in kept_lines) / len(kept_lines)
            except Exception as exc:
                ocr_error = str(exc)
                verbose_logger(
                    f"OCR failed at {sample.timestamp:.1f}s; keeping the frame without OCR: {exc}"
                )

        observation = FrameObservation(
            timestamp=sample.timestamp,
            image_path=relative_path,
            change_score=sample.change_score,
            reasons=reasons,
            ocr_text=note_text,
            ocr_lines=[text for text, _ in kept_lines],
            ocr_confidence=confidence,
        )
        observations.append(observation)
        if note_text and confidence is not None:
            notes.append(
                VisualNote(
                    timestamp=sample.timestamp,
                    text=note_text,
                    lines=[text for text, _ in kept_lines],
                    confidence=confidence,
                    image_path=relative_path,
                )
            )

        ocr_rows.append(
            {
                "timestamp": sample.timestamp,
                "image_path": relative_path,
                "raw_lines": [
                    {"text": text, "score": score} for text, score in raw_lines
                ],
                "kept_lines": [
                    {"text": text, "score": score} for text, score in kept_lines
                ],
                "note_text": note_text,
                "error": ocr_error,
            }
        )
        verbose_logger(
            f"Selected frame {index}/{len(selected)} at {sample.timestamp:.1f}s"
        )

    return observations, notes, ocr_rows


def extract_raw_ocr_lines(result: Sequence[Any]) -> list[tuple[str, float]]:
    raw_lines: list[tuple[str, float]] = []
    for item in result:
        try:
            _, text, score = item
        except (TypeError, ValueError):
            continue
        raw_lines.append((normalize_ocr_text(str(text or "")), float(score)))
    return raw_lines


def _write_frame_observations(
    output_path: Path, observations: Sequence[FrameObservation]
) -> None:
    payload = [
        {
            "timestamp": frame.timestamp,
            "image_path": frame.image_path,
            "change_score": frame.change_score,
            "reasons": frame.reasons,
            "ocr_text": frame.ocr_text,
            "ocr_lines": frame.ocr_lines,
            "ocr_confidence": frame.ocr_confidence,
        }
        for frame in observations
    ]
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_visual_notes(output_path: Path, notes: Sequence[VisualNote]) -> None:
    payload = [
        {
            "timestamp": note.timestamp,
            "text": note.text,
            "lines": note.lines,
            "confidence": note.confidence,
            "image_path": note.image_path,
        }
        for note in notes
    ]
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_jsonl(output_path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
