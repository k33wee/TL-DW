from __future__ import annotations

import main


def test_main_delegates_to_local_video_cli(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_main(argv):
        captured["argv"] = argv

    monkeypatch.setattr(main, "local_video_to_doc_main", fake_main)

    main.main(["--video", "demo.mp4"])

    assert captured["argv"] == ["--video", "demo.mp4"]
