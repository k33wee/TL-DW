from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from tl_dw.local_video_to_doc.cli import _resolve_runtime_args
from tl_dw.local_video_to_doc import (
    Paragraph,
    RenderedChapter,
    TranscriptSegment,
    VisualNote,
    align_paragraphs_to_timestamps,
    attach_visual_notes_to_chapters,
    dedupe_visual_notes,
    extract_source_chapters,
    filter_meaningful_ocr_lines,
    render_markdown,
    safe_slug,
    split_paragraphs_into_generated_chapters,
)


def test_safe_slug_normalizes_symbols_and_accents() -> None:
    assert safe_slug("Caffè Demo / Video #1") == "caffe-demo-video-1"


def test_extract_source_chapters_skips_invalid_ranges() -> None:
    raw = [
        {"start_time": "0.0", "end_time": "30.0", "tags": {"title": "Intro"}},
        {"start_time": "45.0", "end_time": "45.0", "tags": {"title": "Bad"}},
        {"start_time": "60.0", "end_time": "90.0", "tags": {}},
    ]

    chapters = extract_source_chapters(raw, duration=120.0)

    assert [(chapter.title, chapter.start, chapter.end) for chapter in chapters] == [
        ("Intro", 0.0, 30.0),
        ("Chapter 3", 60.0, 90.0),
    ]


def test_align_paragraphs_to_timestamps_preserves_start_times() -> None:
    segments = [
        TranscriptSegment(start=0.0, end=3.0, text=" Hello world. "),
        TranscriptSegment(start=3.0, end=6.0, text="This is the second sentence."),
    ]
    raw_paragraphs = [[" Hello world. "], ["This is the second sentence."]]

    paragraphs = align_paragraphs_to_timestamps(segments, raw_paragraphs)

    assert [(paragraph.start, paragraph.text) for paragraph in paragraphs] == [
        (0.0, "Hello world."),
        (3.0, "This is the second sentence."),
    ]


def test_split_paragraphs_into_generated_chapters_uses_gap_and_duration() -> None:
    paragraphs = [
        Paragraph(start=0.0, text="Opening paragraph."),
        Paragraph(start=120.0, text="Still in the first block."),
        Paragraph(start=620.0, text="New topic starts here."),
    ]

    chapters = split_paragraphs_into_generated_chapters(
        paragraphs,
        max_chapter_seconds=480.0,
        min_gap_seconds=18.0,
    )

    assert len(chapters) == 2
    assert chapters[0].paragraphs[0].text == "Opening paragraph."
    assert chapters[1].paragraphs[0].text == "New topic starts here."


def test_render_markdown_includes_toc_and_timestamps() -> None:
    chapters = [
        RenderedChapter(
            title="Intro",
            start=0.0,
            paragraphs=[Paragraph(start=0.0, text="Welcome.")],
        ),
        RenderedChapter(
            title="Details",
            start=65.0,
            paragraphs=[Paragraph(start=65.0, text="More detail here.")],
        ),
    ]

    markdown = render_markdown(
        title="Demo",
        source_path=Path("demo.mp4"),
        chapters=chapters,
        timestamp_paragraphs=True,
        add_table_of_contents=True,
    )

    assert "### Table of contents" in markdown
    assert "[Intro](#intro)" in markdown
    assert "(00:01:05) More detail here." in markdown


def test_filter_meaningful_ocr_lines_keeps_meeting_context() -> None:
    lines = [
        ("Google Meet", 0.98),
        ("12:34", 0.99),
        ("Participants", 0.88),
        ("..", 0.91),
    ]

    kept = filter_meaningful_ocr_lines(lines, min_score=0.55, min_chars=6)

    assert [text for text, _ in kept] == ["Google Meet", "Participants"]


def test_dedupe_visual_notes_removes_nearby_repeated_frames() -> None:
    notes = [
        VisualNote(
            timestamp=0.0,
            text="Google Meet | Participants",
            lines=["Google Meet"],
            confidence=0.9,
        ),
        VisualNote(
            timestamp=8.0,
            text="Google Meet | Participants",
            lines=["Google Meet"],
            confidence=0.9,
        ),
        VisualNote(
            timestamp=60.0,
            text="Agenda | Sprint Review",
            lines=["Agenda"],
            confidence=0.9,
        ),
    ]

    deduped = dedupe_visual_notes(notes, dedupe_window_sec=45.0)

    assert [note.text for note in deduped] == [
        "Google Meet | Participants",
        "Agenda | Sprint Review",
    ]


def test_attach_visual_notes_to_chapters_groups_by_timestamp() -> None:
    chapters = [
        RenderedChapter(
            title="Part 1",
            start=0.0,
            paragraphs=[Paragraph(start=0.0, text="Intro")],
        ),
        RenderedChapter(
            title="Part 2",
            start=300.0,
            paragraphs=[Paragraph(start=300.0, text="Deep dive")],
        ),
    ]
    notes = [
        VisualNote(
            timestamp=10.0,
            text="Google Meet",
            lines=["Google Meet"],
            confidence=0.9,
        ),
        VisualNote(timestamp=320.0, text="Agenda", lines=["Agenda"], confidence=0.8),
    ]

    attached = attach_visual_notes_to_chapters(chapters, notes)

    assert [note.text for note in attached[0].visual_notes] == ["Google Meet"]
    assert [note.text for note in attached[1].visual_notes] == ["Agenda"]


def test_resolve_runtime_args_maps_gpu_to_cuda_defaults() -> None:
    args = Namespace(device="gpu", whisper_device=None, whisper_compute_type=None)

    _resolve_runtime_args(args)

    assert args.whisper_device == "cuda"
    assert args.whisper_compute_type == "int8_float16"
