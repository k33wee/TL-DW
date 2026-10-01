from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from PIL import Image

from tl_dw.local_video_to_doc.analysis_bundle import build_analysis_sections
from tl_dw.local_video_to_doc.cli import _resolve_runtime_args, build_parser
from tl_dw.local_video_to_doc.frames import (
    SampledFrame,
    _materialize_selected_frames,
    frame_change_score,
    select_sampled_frames,
)
from tl_dw.local_video_to_doc.media import (
    build_audio_extraction_command,
    build_initial_prompt,
    choose_transcription_chapters,
    summary_markdown_path,
)
from tl_dw.local_video_to_doc.models import (
    FrameObservation,
    MediaChapter,
    MediaInfo,
    TranscriptChapter,
)
from tl_dw.local_video_to_doc.ocr_worker import _serializable_result
from tl_dw.local_video_to_doc.transcription import resolve_runtime_settings
from tl_dw.local_video_to_doc.text_utils import build_summary_title
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


def test_transcription_chapters_cover_gaps_without_overlap() -> None:
    media_info = MediaInfo(
        title="Meeting",
        duration=40.0,
        chapters=[
            MediaChapter(title="Intro", start=10.0, end=22.0),
            MediaChapter(title="Details", start=20.0, end=30.0),
        ],
        source_path=Path("meeting.mp4"),
    )

    chapters, used_source_chapters = choose_transcription_chapters(
        media_info, ignore_source_chapters=False
    )

    assert used_source_chapters is True
    assert [(chapter.title, chapter.start, chapter.end) for chapter in chapters] == [
        ("Transcript", 0.0, 10.0),
        ("Intro", 10.0, 22.0),
        ("Details", 22.0, 30.0),
        ("Transcript", 30.0, 40.0),
    ]


def test_initial_prompt_ignores_opaque_recording_name_but_keeps_real_titles() -> None:
    chapters = [
        MediaChapter(title="Transcript", start=0.0, end=10.0),
        MediaChapter(title="Budget review", start=10.0, end=20.0),
    ]

    generated = build_initial_prompt(
        "ScreenRec-2026-09-30-16.04.32",
        chapters,
        source_name="ScreenRec-2026-09-30-16.04.32.mp4",
    )
    descriptive = build_initial_prompt(
        "Quarterly Planning",
        [],
        source_name="Quarterly_Planning.mp4",
    )
    embedded = build_initial_prompt(
        "Quarterly planning",
        [],
        source_name="recording.mp4",
    )
    override = build_initial_prompt(
        "ScreenRec-2026-09-30-16.04.32",
        [],
        "  tariff scaleId  ",
        source_name="ScreenRec-2026-09-30-16.04.32.mp4",
    )

    assert generated == "Budget review"
    assert descriptive == "Quarterly Planning"
    assert embedded == "Quarterly planning"
    assert override == "tariff scaleId"


def test_summary_title_uses_recurring_topic_instead_of_filename() -> None:
    texts = [
        "Allora, buongiorno a tutti.",
        "Oggi parliamo delle scale tariffarie legacy e della modalita griglia.",
        "La scala chilometrica non puo essere normalizzata sul contratto.",
        "Poi c'e un errore nel caricamento.",
        "Le scale tariffarie legacy restano legate al contratto e alla modalita griglia.",
    ]

    title = build_summary_title(
        texts,
        fallback="Screen Recording 2026-10-01 at 11.55.49",
    )

    assert "tariffarie" in title.lower()
    assert "griglia" in title.lower()
    assert not title.lower().startswith("allora")
    assert not title.lower().startswith("oggi")
    assert "screen recording" not in title.lower()
    assert len(title) <= 90


def test_summary_title_prefers_recurring_topics_over_a_spoken_fragment() -> None:
    texts = [
        "Questi due campi sono il service type e il track e devono puntare alla stessa cosa.",
        "Il service type bianco diventa tracking e anche invoicing segue lo stesso track.",
        "Poi il problema delle scale sulla tariffa reale: la scala chilometrica sembra fixed.",
        "Le scale della tariffa non dicono quale size caricare.",
        "Ci sono tre scale in una tariffa real e non viene detto quale e da gestire.",
        "Il frontend e il backend devono salvare la scala scelta sulla tariffa.",
    ] * 3

    title = build_summary_title(texts, fallback="Screen Recording")

    lowered = title.lower()
    assert "service type" in lowered
    assert "scale" in lowered or "scala" in lowered
    assert "tariff" in lowered
    assert "non viene detto" not in lowered
    assert len(title) <= 90


def test_summary_title_falls_back_when_transcript_has_no_topic() -> None:
    assert build_summary_title([], "Quarterly planning") == "Quarterly planning"
    assert build_summary_title(["Ok.", "Ciao a tutti."], "Fallback meeting") == (
        "Fallback meeting"
    )


def test_summary_markdown_path_names_default_file_and_keeps_explicit_path() -> None:
    artifact_dir = Path("output/meeting")

    generated = summary_markdown_path(
        artifact_dir / "document.md",
        artifact_dir,
        "Scale tariffarie legacy e modalita griglia",
    )
    explicit = summary_markdown_path(
        Path("notes/custom-name.md"),
        artifact_dir,
        "Scale tariffarie legacy e modalita griglia",
    )

    assert generated == artifact_dir / "scale-tariffarie-legacy-e-modalita-griglia.md"
    assert explicit == Path("notes/custom-name.md")


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


def test_cli_quality_defaults_enable_large_model_normalization_and_frames(
    monkeypatch,
) -> None:
    for name in [
        "EXTRACTION_WHISPER_MODEL",
        "EXTRACTION_NORMALIZE_AUDIO",
        "EXTRACTION_DISABLE_FRAMES",
        "EXTRACTION_DISABLE_OCR",
        "EXTRACTION_SECTION_SECONDS",
        "EXTRACTION_MAX_FRAMES_PER_SECTION",
        "EXTRACTION_SAT_MODEL",
    ]:
        monkeypatch.delenv(name, raising=False)
    args = build_parser().parse_args(["--video", "meeting.mp4"])

    assert args.whisper_model == "large-v3"
    assert args.normalize_audio is True
    assert args.use_frames is True
    assert args.use_ocr is True
    assert args.section_seconds == 120
    assert args.max_frames_per_section == 4
    assert args.sat_model == "simple"


def test_audio_extraction_command_applies_merged_normalization_filter() -> None:
    command = build_audio_extraction_command(
        Path("meeting.mp4"),
        Path("audio.wav"),
        normalize_audio=True,
        normalization_filter="highpass=f=80,loudnorm=I=-16",
    )

    assert command[:6] == [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    assert command[command.index("-af") + 1] == "highpass=f=80,loudnorm=I=-16"
    assert command[-5:] == [
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "audio.wav",
    ]


def test_audio_extraction_command_can_preserve_clean_source_audio() -> None:
    command = build_audio_extraction_command(
        Path("meeting.mp4"),
        Path("audio.wav"),
        normalize_audio=False,
        normalization_filter="unused",
    )

    assert "-af" not in command
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"


def test_runtime_profiles_include_macbook_and_pascal_safe_defaults() -> None:
    macbook = resolve_runtime_settings("macbook")
    pascal = resolve_runtime_settings("cuda-pascal")

    assert (macbook.device, macbook.compute_type) == ("cpu", "int8")
    assert (pascal.device, pascal.compute_type) == ("cuda", "int8")


def test_frame_change_score_detects_a_screen_transition() -> None:
    black = Image.new("L", (256, 144), color=0)
    white = Image.new("L", (256, 144), color=255)

    assert frame_change_score(black, black) == 0
    assert frame_change_score(black, white) > 0.5


def test_frame_selection_keeps_section_anchors_and_strong_changes() -> None:
    sampled = [
        SampledFrame(0.0, Path("0.jpg"), 0.0),
        SampledFrame(5.0, Path("5.jpg"), 0.001),
        SampledFrame(30.0, Path("30.jpg"), 0.25),
        SampledFrame(90.0, Path("90.jpg"), 0.08),
        SampledFrame(120.0, Path("120.jpg"), 0.01),
        SampledFrame(180.0, Path("180.jpg"), 0.3),
    ]

    selected = select_sampled_frames(
        sampled,
        media_duration=240.0,
        section_sec=120.0,
        max_frames_per_section=3,
        min_change_score=0.015,
        min_spacing_sec=5.0,
    )

    assert [frame.timestamp for frame, _ in selected] == [0.0, 30.0, 90.0, 120.0, 180.0]
    assert selected[0][1] == ["section-anchor"]
    assert "scene-change" in selected[1][1]
    assert "section-anchor" in selected[3][1]


def test_frame_is_kept_when_optional_ocr_fails(tmp_path) -> None:
    source = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 18), color="white").save(source)
    artifact_dir = tmp_path / "artifacts"
    frames_dir = artifact_dir / "frames"
    frames_dir.mkdir(parents=True)
    messages: list[str] = []

    def failing_ocr(_path):
        raise RuntimeError("synthetic OCR failure")

    observations, notes, rows = _materialize_selected_frames(
        selected=[(SampledFrame(5.0, source, 0.2), ["scene-change"])],
        frames_dir=frames_dir,
        artifact_dir=artifact_dir,
        ocr_engine=failing_ocr,
        ocr_min_score=0.5,
        ocr_min_chars=3,
        ocr_max_lines=3,
        ocr_max_note_chars=100,
        verbose_logger=messages.append,
    )

    assert len(observations) == 1
    assert (artifact_dir / observations[0].image_path).is_file()
    assert observations[0].ocr_text == ""
    assert notes == []
    assert rows[0]["raw_lines"] == []
    assert any("keeping the frame without OCR" in message for message in messages)


def test_ocr_worker_discards_nonserializable_boxes() -> None:
    result = _serializable_result(
        [
            [[[0, 0], [1, 1]], "Visible text", 0.95],
            [None, "Second line", 0.75],
        ]
    )

    assert result == [
        [None, "Visible text", 0.95],
        [None, "Second line", 0.75],
    ]


def test_analysis_sections_align_transcript_and_images_without_loss() -> None:
    transcript = [
        TranscriptChapter(
            title="Meeting",
            segments=[
                TranscriptSegment(0.0, 10.0, " Opening"),
                TranscriptSegment(119.0, 121.0, " Boundary topic"),
                TranscriptSegment(200.0, 210.0, " Follow-up"),
            ],
        )
    ]
    frames = [
        FrameObservation(0.0, "frames/a.jpg", 0.0, ["section-anchor"]),
        FrameObservation(120.0, "frames/b.jpg", 0.2, ["scene-change"]),
    ]

    sections = build_analysis_sections(
        transcript,
        frames,
        media_duration=240.0,
        section_seconds=120.0,
    )

    assert len(sections) == 2
    assert [row["text"] for row in sections[0]["transcript"]] == ["Opening"]
    assert [row["text"] for row in sections[1]["transcript"]] == [
        "Boundary topic",
        "Follow-up",
    ]
    assert sections[0]["frames"][0]["image_path"] == "frames/a.jpg"
    assert sections[1]["frames"][0]["image_path"] == "frames/b.jpg"
