from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "videos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
