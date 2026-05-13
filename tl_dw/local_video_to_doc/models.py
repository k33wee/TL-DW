from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MediaChapter:
    title: str
    start: float
    end: float


@dataclass
class MediaInfo:
    title: str
    duration: float
    chapters: list[MediaChapter]
    source_path: Path


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptChapter:
    title: str
    segments: list[TranscriptSegment]


@dataclass
class Paragraph:
    start: float
    text: str


@dataclass
class VisualNote:
    timestamp: float
    text: str
    lines: list[str]
    confidence: float


@dataclass
class RenderedChapter:
    title: str
    start: float
    paragraphs: list[Paragraph]
    visual_notes: list[VisualNote] = field(default_factory=list)


@dataclass
class OutputPaths:
    artifact_dir: Path
    document_path: Path
