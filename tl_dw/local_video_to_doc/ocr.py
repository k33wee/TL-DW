from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import VisualNote
from .ocr_filters import (
    dedupe_visual_notes,
    filter_meaningful_ocr_lines,
    normalize_ocr_text,
    summarize_visual_lines,
)


def load_ocr_engine() -> Any:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'rapidocr-onnxruntime'. Install project dependencies before running OCR extraction."
        ) from exc

    return RapidOCR()


def extract_visual_notes(
    video_path: Path,
    media_duration: float,
    ocr_engine: Any,
    sample_sec: float,
    min_score: float,
    min_chars: int,
    max_lines: int,
    max_note_chars: int,
    dedupe_window_sec: float,
    artifact_dir: Path,
    verbose_logger: callable,
) -> list[VisualNote]:
    if sample_sec <= 0:
        raise SystemExit("--ocr-sample-sec must be greater than 0.")

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'opencv-python'. Install project dependencies before OCR extraction."
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video for OCR sampling: {video_path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration = (
            media_duration
            if media_duration > 0
            else (frame_count / fps if fps > 0 else 0.0)
        )
        timestamps = _build_sample_timestamps(duration=duration, sample_sec=sample_sec)
        notes = _collect_visual_notes(
            capture=capture,
            ocr_engine=ocr_engine,
            timestamps=timestamps,
            min_score=min_score,
            min_chars=min_chars,
            max_lines=max_lines,
            max_note_chars=max_note_chars,
            artifact_dir=artifact_dir,
            verbose_logger=verbose_logger,
        )
    finally:
        capture.release()

    deduped_notes = dedupe_visual_notes(notes, dedupe_window_sec=dedupe_window_sec)
    _write_visual_notes(artifact_dir / "visual_notes.json", deduped_notes)
    return deduped_notes


def _build_sample_timestamps(duration: float, sample_sec: float) -> list[float]:
    if duration <= 0:
        return [0.0]

    steps = max(1, int(duration // sample_sec) + 1)
    timestamps = [round(index * sample_sec, 3) for index in range(steps)]
    if timestamps[-1] < duration:
        timestamps.append(round(duration, 3))
    return timestamps


def _collect_visual_notes(
    capture: Any,
    ocr_engine: Any,
    timestamps: list[float],
    min_score: float,
    min_chars: int,
    max_lines: int,
    max_note_chars: int,
    artifact_dir: Path,
    verbose_logger: callable,
) -> list[VisualNote]:
    notes: list[VisualNote] = []
    ocr_jsonl = artifact_dir / "ocr.jsonl"
    with ocr_jsonl.open("w", encoding="utf-8") as jsonl_file:
        for timestamp in timestamps:
            note = _extract_note_for_timestamp(
                capture=capture,
                ocr_engine=ocr_engine,
                timestamp=timestamp,
                min_score=min_score,
                min_chars=min_chars,
                max_lines=max_lines,
                max_note_chars=max_note_chars,
                jsonl_file=jsonl_file,
            )
            if note is not None:
                notes.append(note)
                if len(notes) % 10 == 0:
                    verbose_logger(f"OCR notes kept: {len(notes)}")
    return notes


def _extract_note_for_timestamp(
    capture: Any,
    ocr_engine: Any,
    timestamp: float,
    min_score: float,
    min_chars: int,
    max_lines: int,
    max_note_chars: int,
    jsonl_file: Any,
) -> VisualNote | None:
    import cv2

    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = capture.read()
    if not ok:
        return None

    result, _ = ocr_engine(frame)
    raw_lines = _extract_raw_lines(result or [])
    kept = filter_meaningful_ocr_lines(
        raw_lines, min_score=min_score, min_chars=min_chars
    )
    note_lines = [text for text, _ in kept]
    note_text = summarize_visual_lines(
        note_lines, max_lines=max_lines, max_chars=max_note_chars
    )

    row = {
        "timestamp": timestamp,
        "raw_lines": [{"text": text, "score": score} for text, score in raw_lines],
        "kept_lines": [{"text": text, "score": score} for text, score in kept],
        "note_text": note_text,
    }
    jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not note_text:
        return None

    confidence = sum(score for _, score in kept) / len(kept)
    return VisualNote(
        timestamp=timestamp, text=note_text, lines=note_lines, confidence=confidence
    )


def _extract_raw_lines(result: list[Any]) -> list[tuple[str, float]]:
    raw_lines: list[tuple[str, float]] = []
    for item in result:
        try:
            _, text, score = item
        except Exception:
            continue
        raw_lines.append((normalize_ocr_text(text or ""), float(score)))
    return raw_lines


def _write_visual_notes(output_path: Path, notes: list[VisualNote]) -> None:
    payload = [
        {
            "timestamp": note.timestamp,
            "text": note.text,
            "lines": note.lines,
            "confidence": note.confidence,
        }
        for note in notes
    ]
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
