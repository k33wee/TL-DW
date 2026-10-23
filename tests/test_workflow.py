from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from PIL import Image

from tl_dw.local_video_to_doc import pipeline
from tl_dw.local_video_to_doc.artifacts import PreparationCache, preparation_lock
from tl_dw.local_video_to_doc.cli import _resolve_runtime_args, build_parser, main
from tl_dw.local_video_to_doc.documents import (
    _validate_container,
    prepare_documents,
    resolve_documents,
)
from tl_dw.local_video_to_doc.fallback_segmenter import SimpleParagraphSegmenter
from tl_dw.local_video_to_doc.media import (
    extract_audio,
    ffprobe_metadata,
    resolve_videos,
)
from tl_dw.local_video_to_doc.models import TranscriptChapter, TranscriptSegment
from tl_dw.local_video_to_doc.preflight import (
    media_preflight,
    require_audio_selection,
    select_audio,
)


def test_input_directory_requires_exactly_one_video(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"video")
    assert resolve_videos(tmp_path, tmp_path) == [tmp_path / "a.mp4"]
    (tmp_path / "b.MOV").write_bytes(b"video")
    with pytest.raises(SystemExit, match="exactly one"):
        resolve_videos(tmp_path, tmp_path)


def test_cache_checks_content_and_never_accepts_running_stage(tmp_path):
    output = tmp_path / "artifact.txt"
    output.write_text("first")
    cache = PreparationCache(tmp_path)
    cache.begin("asr", "key")
    assert not cache.reusable("asr", "key")
    cache.complete("asr", "key", [output])
    assert PreparationCache(tmp_path).reusable("asr", "key")
    assert not cache.reusable("asr", "new-key")
    assert not PreparationCache(tmp_path, force=True).reusable("asr", "key")
    output.write_text("corrupted")
    assert not cache.reusable("asr", "key")


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"schema_version": 1, "stages": []},
        {"schema_version": 1, "stages": {"asr": []}},
    ],
)
def test_malformed_preparation_cache_is_not_trusted(tmp_path, payload):
    (tmp_path / "preparation.json").write_text(json.dumps(payload))
    cache = PreparationCache(tmp_path)
    assert not cache.reusable("asr", "key")
    cache.begin("asr", "key")


def test_preparation_has_an_exclusive_crash_safe_os_lock(tmp_path):
    with preparation_lock(tmp_path), pytest.raises(RuntimeError, match="Another"):
        with preparation_lock(tmp_path):
            pytest.fail("lock was not exclusive")
    with preparation_lock(tmp_path):
        pass


def test_preflight_space_failure_is_before_models(tmp_path, monkeypatch):
    source = tmp_path / "a.mp4"
    source.write_bytes(b"x")
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: SimpleNamespace(free=0))
    args = build_parser().parse_args([])
    metadata = {
        "format": {"duration": "60"},
        "streams": [
            {"index": 0, "codec_type": "video"},
            {"index": 1, "codec_type": "audio"},
        ],
    }
    with pytest.raises(RuntimeError, match="Insufficient disk"):
        media_preflight(source, metadata, tmp_path, args)
    report = json.loads((tmp_path / "preflight.json").read_text())
    assert report["estimated_temporary_bytes"] > 0
    assert "Not applied" in report["compression"]


def test_ambiguous_and_sampled_quiet_tracks_are_not_silently_discarded(
    tmp_path, monkeypatch
):
    from tl_dw.local_video_to_doc import preflight

    tracks = [
        dict(
            index=i, codec_type="audio", codec_name="pcm", channels=1, sample_rate=16000
        )
        for i in [1, 2]
    ]

    def diagnostic(_video, stream, _duration, scope):
        return dict(
            index=stream["index"],
            codec_name="pcm",
            channels=1,
            sample_rate=16000,
            payload_sha256=str(stream["index"]),
            silent_full_track=False,
            quiet_sample_only=True,
            volume_scope=scope,
        )

    monkeypatch.setattr(preflight, "diagnose_track", diagnostic)
    result = select_audio(
        tmp_path / "test.mp4",
        {"format": {"duration": "3600"}, "streams": tracks},
        scope="sample",
    )
    assert result["mode"] == "ambiguous"
    with pytest.raises(ValueError, match="explicitly"):
        require_audio_selection(result)
    explicit = select_audio(Path("not-read"), {"streams": tracks}, mix=[1, 2])
    assert explicit["selected_streams"] == [1, 2]
    with pytest.raises(ValueError, match="distinct"):
        select_audio(Path("not-read"), {"streams": tracks}, mix=[1, 1])


def test_equal_audio_packets_with_different_offsets_are_not_deduplicated(
    tmp_path, monkeypatch
):
    from tl_dw.local_video_to_doc import preflight

    streams = [
        dict(
            index=i,
            codec_type="audio",
            codec_name="pcm",
            channels=1,
            sample_rate=16000,
            start_time=str(i),
        )
        for i in (1, 2)
    ]
    monkeypatch.setattr(
        preflight,
        "diagnose_track",
        lambda _v, stream, _d, _s: {
            **stream,
            "payload_sha256": "same",
            "silent_full_track": False,
        },
    )
    result = select_audio(
        tmp_path / "not-read", {"format": {"duration": "2"}, "streams": streams}
    )
    assert result["mode"] == "ambiguous"
    assert result["candidate_streams"] == [1, 2]


@pytest.fixture
def synthetic_multitrack(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg unavailable")
    video = tmp_path / "synthetic.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=64x64:r=2:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=1",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-map",
            "1:a",
            "-map",
            "2:a",
            "-t",
            "1",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            str(video),
        ],
        check=True,
    )
    return video


def test_real_ffmpeg_deduplicates_full_tracks_and_excludes_full_silence(
    synthetic_multitrack, tmp_path
):
    selection = select_audio(
        synthetic_multitrack, ffprobe_metadata(synthetic_multitrack)
    )
    assert selection["selected_streams"] == [1]
    assert selection["tracks"][1]["duplicate_of"] == 1
    assert selection["tracks"][2]["silent_full_track"] is True
    assert selection["diagnostic_scope"] == "full"
    output = tmp_path / "mixed.wav"
    extract_audio(
        synthetic_multitrack, output, audio_streams=[1, 2], normalize_audio=False
    )
    import wave

    with wave.open(str(output)) as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 16000
        assert audio.getnframes() > 10000


def test_real_media_pipeline_and_resume_without_real_asr_model(
    synthetic_multitrack, tmp_path, monkeypatch
):
    from tl_dw.local_video_to_doc import cli

    loads = []

    def load(**_kwargs):
        loads.append("whisper")
        return SimpleNamespace(
            transcribe=lambda *_a, **_k: (
                [SimpleNamespace(start=0.0, end=0.5, text="Synthetic meeting fixture")],
                SimpleNamespace(language="en"),
            )
        )

    monkeypatch.setattr(cli, "load_whisper_model", load)
    output = tmp_path / "complete"
    document = tmp_path / "specification.docx"
    _docx(document)
    args = [
        "--video",
        str(synthetic_multitrack),
        "--output",
        str(output),
        "--no-ocr",
        "--attachment",
        str(document),
    ]
    cli.main(args)
    root = output / "synthetic"
    bundle = json.loads((root / "analysis.json").read_text())
    assert bundle["stats"]["selected_frames"] >= 1
    assert bundle["stats"]["transcript_segments"] == 1
    assert (
        json.loads((root / "preparation.json").read_text())["stages"]["documents"][
            "status"
        ]
        == "complete"
    )
    cli.main(args)
    assert loads == ["whisper"]
    document = tmp_path / "specification.docx"
    _docx(document)
    cli.main(["--bundle", str(root), "--attachment", str(document)])
    context = json.loads((root / "documents.json").read_text())
    assert context["unique_figures"] == 1
    assert context["analysis_sha256"]
    assert loads == ["whisper"]


def test_selected_late_starting_audio_retains_video_timestamp_origin(tmp_path):
    import array
    import wave

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg unavailable")
    video = tmp_path / "late.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=64x64:r=2:d=2",
            "-itsoffset",
            "1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=1",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            str(video),
        ],
        check=True,
    )
    output = tmp_path / "aligned.wav"
    extract_audio(video, output, normalize_audio=False, audio_streams=[1])
    with wave.open(str(output)) as audio:
        samples = array.array("h", audio.readframes(audio.getnframes()))
    assert len(samples) >= 30000
    assert max(abs(value) for value in samples[:8000]) == 0
    assert max(abs(value) for value in samples[20000:]) > 100


def _docx(path):
    image = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(image, format="PNG")
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        )
        archive.writestr(
            "_rels/.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>',
        )
        archive.writestr(
            "word/_rels/document.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/diagram.png"/></Relationships>',
        )
        archive.writestr(
            "word/document.xml",
            """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><w:body><w:p><w:r><w:t>Confirmed specification, not a live deployment.</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Rule</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc></w:tr></w:tbl><w:p><w:r><w:drawing><a:blip r:embed="rId1"/></w:drawing></w:r></w:p></w:body></w:document>""",
        )
        archive.writestr("word/media/diagram.png", image.getvalue())


def test_context_docx_preserves_text_table_and_figure_locally(tmp_path):
    source = tmp_path / "L'ordine — specifica.docx"
    _docx(source)
    output = tmp_path / "artifacts"
    output.mkdir()
    result = prepare_documents([source], output)
    document = result["documents"][0]
    text = (output / document["markdown_path"]).read_text()
    assert "Confirmed specification" in text
    assert "Rule" in text and "Value" in text
    assert len(document["figures"]) == 1
    assert (output / document["figures"][0]["image_path"]).is_file()
    assert result == prepare_documents([source], output)
    figure = output / document["figures"][0]["image_path"]
    original = figure.read_bytes()
    figure.write_bytes(b"corrupted")
    assert result == prepare_documents([source], output)
    assert figure.read_bytes() == original
    assert result["conversion"] == "local; hosted OCR rejected"


def test_context_directory_bounds_deduplication_and_text_changes(tmp_path):
    source = tmp_path / "context"
    source.mkdir()
    text = source / "spec.txt"
    text.write_text("A proposed feature")
    (source / "video.mp4").write_bytes(b"do not read")
    assert resolve_documents([source, text]) == [text]
    output = tmp_path / "output"
    output.mkdir()
    first = prepare_documents([text], output)
    text.write_text("A revised proposed feature")
    second = prepare_documents([text], output)
    assert first["signature"] != second["signature"]
    assert (
        first["documents"][0]["markdown_path"]
        != second["documents"][0]["markdown_path"]
    )
    text.write_text("x" * 200001)
    with pytest.raises(ValueError, match="Context exceeds"):
        prepare_documents([text], output)


@pytest.mark.parametrize(
    "member", ["../outside", "/tmp/outside", "word/../../outside", "C:\\outside"]
)
def test_unsafe_archives_are_rejected_without_extracting(member):
    data = io.BytesIO()
    with ZipFile(data, "w") as archive:
        archive.writestr(member, "malicious")
    with pytest.raises(ValueError, match="Unsafe"):
        _validate_container(data.getvalue())


def _fake_pipeline(tmp_path, monkeypatch):
    video = tmp_path / "meeting.mp4"
    video.write_bytes(b"synthetic source")
    args = build_parser().parse_args(
        ["--video", str(video), "--output", str(tmp_path / "output"), "--no-frames"]
    )
    _resolve_runtime_args(args)
    counts = {"audio": 0, "asr": 0, "resources": []}
    monkeypatch.setattr(pipeline, "require_media_tools", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "ffprobe_metadata",
        lambda _p: {
            "format": {"duration": "2"},
            "streams": [
                {"index": 0, "codec_type": "video"},
                {"index": 1, "codec_type": "audio"},
            ],
        },
    )

    def extract(_video, target, **_kwargs):
        counts["audio"] += 1
        target.write_bytes(b"synthetic PCM")

    def transcribe(**_kwargs):
        counts["asr"] += 1
        return [
            TranscriptChapter(
                "Transcript",
                [TranscriptSegment(0, 1, f"Meeting revision {counts['asr']}")],
            )
        ], {}

    def resource(name):
        counts["resources"].append(name)
        return SimpleParagraphSegmenter() if name == "sat" else object()

    monkeypatch.setattr(pipeline, "extract_audio", extract)
    monkeypatch.setattr(pipeline, "transcribe_media", transcribe)

    def run():
        pipeline.process_video(video, args, None, None, None, False, resource)

    return run, args, counts, tmp_path / "output/meeting"


def test_legacy_bundle_is_preserved_without_silently_retranscribing(
    tmp_path, monkeypatch
):
    run, args, counts, root = _fake_pipeline(tmp_path, monkeypatch)
    root.mkdir(parents=True)
    (root / "analysis.json").write_text('{"legacy":"preserve"}')
    with pytest.raises(ValueError, match="legacy"):
        run()
    assert counts == {"audio": 0, "asr": 0, "resources": []}
    assert "preserve" in (root / "analysis.json").read_text()
    args.force_preparation = True
    run()
    assert counts["asr"] == 1


def test_resume_reuses_all_stages_without_loading_models(tmp_path, monkeypatch):
    run, args, counts, root = _fake_pipeline(tmp_path, monkeypatch)
    run()
    original = (root / "analysis.json").read_bytes()
    counts["resources"].clear()
    run()
    assert counts == {"audio": 1, "asr": 1, "resources": []}
    assert (root / "analysis.json").read_bytes() == original
    invocation = json.loads((root / "provenance.json").read_text())["this_invocation"]
    assert invocation["executed_stages"] == []
    assert "transcription" in invocation["reused_stages"]
    args.whisper_language = "it"
    run()
    assert counts["audio"] == 1 and counts["asr"] == 2
    assert "revision 2" in (root / "analysis.json").read_text()


def test_cancellation_after_asr_resumes_without_retranscribing(tmp_path, monkeypatch):
    run, _args, counts, root = _fake_pipeline(tmp_path, monkeypatch)
    original = pipeline._extract_visual_context
    monkeypatch.setattr(
        pipeline,
        "_extract_visual_context",
        lambda *_a: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        run()
    assert counts["asr"] == 1
    assert (
        json.loads((root / "preparation.json").read_text())["stages"]["visuals"][
            "status"
        ]
        == "running"
    )
    monkeypatch.setattr(pipeline, "_extract_visual_context", original)
    run()
    assert counts["asr"] == 1
    assert (root / "analysis.json").is_file()


def test_corrupted_transcript_is_rebuilt_and_updates_bundle(tmp_path, monkeypatch):
    run, _args, counts, root = _fake_pipeline(tmp_path, monkeypatch)
    run()
    (root / "transcript.txt").write_text("corrupted")
    run()
    assert counts["asr"] == 2
    assert "revision 2" in (root / "analysis.json").read_text()


def test_preflight_only_never_loads_models_or_extracts(tmp_path, monkeypatch):
    run, args, counts, root = _fake_pipeline(tmp_path, monkeypatch)
    args.preflight_only = True
    run()
    assert counts == {"audio": 0, "asr": 0, "resources": []}
    assert (root / "preflight.json").is_file()
    assert not (root / "audio.wav").exists()


def test_context_is_bound_to_prepared_bundle_and_transparency_is_preserved(tmp_path):
    source = tmp_path / "diagram.png"
    Image.new("RGBA", (16, 16), (0, 0, 0, 0)).save(source)
    root = tmp_path / "output"
    root.mkdir()
    (root / "analysis.json").write_text('{"revision":1}')
    first = prepare_documents([source], root)
    figure = first["documents"][0]["figures"][0]
    with Image.open(root / figure["image_path"]) as image:
        assert image.getpixel((0, 0)) == (255, 255, 255)
    (root / "analysis.json").write_text('{"revision":2}')
    second = prepare_documents([source], root)
    assert first["analysis_sha256"] != second["analysis_sha256"]
    assert first["signature"] != second["signature"]


def test_document_conversion_never_enables_hosted_ocr(tmp_path, monkeypatch):
    import anydoc

    source = tmp_path / "scan.pdf"
    source.write_bytes(b"%PDF synthetic")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "synthetic-never-send")

    def reject_scan(_data, _format, **kwargs):
        assert kwargs["ocr"] == "reject"
        raise ValueError("scanned PDF requires OCR")

    monkeypatch.setattr(anydoc, "to_markdown_bytes", reject_scan)
    root = tmp_path / "output"
    root.mkdir()
    with pytest.raises(ValueError, match="requires OCR"):
        prepare_documents([source], root)
    assert not (root / "documents.json").exists()


def test_preparation_rejects_symlink_outputs(tmp_path, monkeypatch):
    run, _args, _counts, root = _fake_pipeline(tmp_path, monkeypatch)
    root.mkdir(parents=True)
    outside = tmp_path / "preserve.wav"
    outside.write_bytes(b"must not overwrite")
    (root / "audio.wav").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        run()
    assert outside.read_bytes() == b"must not overwrite"


def test_bundle_attachments_only_never_load_asr(tmp_path, monkeypatch):
    source = tmp_path / "spec.txt"
    source.write_text("Some requirements")
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "analysis.json").write_text("{}")
    from tl_dw.local_video_to_doc import cli

    monkeypatch.setattr(
        cli, "load_whisper_model", lambda **_k: pytest.fail("ASR model loaded")
    )
    main(["--bundle", str(root), "--attachment", str(source)])
    assert (root / "documents.json").is_file()
