"""Local-video to readable Markdown pipeline."""

from .chapters import (
    attach_visual_notes_to_chapters,
    split_paragraphs_into_generated_chapters,
)
from .cli import main
from .media import extract_source_chapters
from .models import Paragraph, RenderedChapter, TranscriptSegment, VisualNote
from .ocr_filters import dedupe_visual_notes, filter_meaningful_ocr_lines
from .paragraphs import align_paragraphs_to_timestamps
from .rendering import render_markdown
from .text_utils import safe_slug

__all__ = [
    "Paragraph",
    "RenderedChapter",
    "TranscriptSegment",
    "VisualNote",
    "align_paragraphs_to_timestamps",
    "attach_visual_notes_to_chapters",
    "dedupe_visual_notes",
    "extract_source_chapters",
    "filter_meaningful_ocr_lines",
    "main",
    "render_markdown",
    "safe_slug",
    "split_paragraphs_into_generated_chapters",
]
