"""Fail packaging checks if runtime resources are missing or local data leaks in."""

from __future__ import annotations

import argparse
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

RUNTIME_FILES = {
    ".pi/extensions/tl-dw.ts",
    ".pi/skills/tl-dw/SKILL.md",
    ".python-version",
    "package.json",
    "pyproject.toml",
    "uv.lock",
    "tl_dw/local_video_to_doc/cli.py",
    "tl_dw/local_video_to_doc/documents.py",
}
REQUIRED = {
    "wheel": {
        "tl_dw/local_video_to_doc/cli.py",
        "tl_dw/local_video_to_doc/documents.py",
    },
    "sdist": RUNTIME_FILES
    | {
        "package-lock.json",
        "tsconfig.json",
        "scripts/check_distributions.py",
        "tests/package.test.ts",
    },
    "npm": RUNTIME_FILES,
}
PRIVATE_PARTS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "output",
    "tl-dw-output",
}
PRIVATE_SUFFIXES = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".wav",
    ".mp3",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".pyc",
    ".jsonl",
    ".doc",
    ".docx",
    ".docm",
    ".pdf",
    ".ppt",
    ".pptx",
    ".pptm",
    ".pps",
    ".ppsx",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".xlsb",
    ".odt",
    ".ods",
    ".odp",
    ".rtf",
    ".epub",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bin",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}


def validate_paths(names: list[str], kind: str) -> int:
    paths = set()
    prefix = None
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Unsafe archive path: {name}")
        if kind != "wheel":
            if not path.parts:
                continue
            prefix = prefix or path.parts[0]
            if path.parts[0] != prefix:
                raise ValueError(f"Multiple archive roots: {name}")
            path = PurePosixPath(*path.parts[1:])
        parts = set(path.parts)
        if (
            parts & PRIVATE_PARTS
            or path.suffix.lower() in PRIVATE_SUFFIXES
            or any(p == ".env" or p.startswith(".env.") for p in parts)
            or path.as_posix().startswith((".pi/memory/", ".pi/local-memory/"))
            or (path.parent == PurePosixPath(".pi") and path.suffix == ".json")
        ):
            raise ValueError(f"Private/local artifact in distribution: {name}")
        paths.add(path.as_posix())
    missing = REQUIRED[kind] - paths
    if missing:
        raise ValueError(
            f"{kind} is missing required sources/resources: {sorted(missing)}"
        )
    return len(paths)


def validate_archive(archive_path: Path) -> tuple[str, int]:
    if archive_path.name.endswith(".whl"):
        kind = "wheel"
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if any(stat.S_ISLNK(member.external_attr >> 16) for member in members):
                raise ValueError("Distribution must not contain symlinks")
            names = [member.filename for member in members if not member.is_dir()]
    else:
        kind = "npm" if archive_path.name.endswith(".tgz") else "sdist"
        with tarfile.open(archive_path) as archive:
            members = archive.getmembers()
            if any(member.issym() or member.islnk() for member in members):
                raise ValueError("Distribution must not contain links")
            names = [member.name for member in members if member.isfile()]
    return kind, validate_paths(names, kind)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    kinds = set()
    for archive in sorted(directory.iterdir()):
        if archive.name.endswith((".whl", ".tar.gz", ".tgz")):
            kind, count = validate_archive(archive)
            kinds.add(kind)
            print(f"{archive.name}: {count} safe source/metadata paths")
    if kinds != set(REQUIRED):
        raise SystemExit(
            f"Expected wheel, sdist and npm archives; found {sorted(kinds)}"
        )


if __name__ == "__main__":
    main()
