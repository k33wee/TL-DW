from __future__ import annotations

from typing import Sequence

from tl_dw.local_video_to_doc.cli import main as local_video_to_doc_main


def main(argv: Sequence[str] | None = None) -> None:
    """Run the main local-video document generation CLI."""
    local_video_to_doc_main(argv)


if __name__ == "__main__":
    main()
