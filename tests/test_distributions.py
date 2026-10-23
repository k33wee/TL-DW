from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from scripts.check_distributions import REQUIRED, validate_archive, validate_paths


@pytest.mark.parametrize("kind", ["wheel", "sdist", "npm"])
def test_distribution_has_required_runtime_and_development_resources(kind):
    prefix = "" if kind == "wheel" else "package/"
    names = [prefix + name for name in REQUIRED[kind]]
    assert validate_paths(names, kind) == len(REQUIRED[kind])


@pytest.mark.parametrize(
    "private",
    [
        ".pi/memory/MEMORY.md",
        ".pi/local-memory/private.md",
        ".pi/settings.json",
        ".env",
        "output/transcript.txt",
        "node_modules/dependency/index.js",
        "recording.MOV",
        "seed.sqlite",
        "context.docx",
        "../outside",
    ],
)
def test_distribution_rejects_local_or_unsafe_files(private):
    names = ["package/" + name for name in REQUIRED["npm"]]
    with pytest.raises(ValueError):
        validate_paths([*names, "package/" + private], "npm")


def test_missing_python_version_file_is_a_packaging_error():
    names = ["package/" + name for name in REQUIRED["npm"] - {".python-version"}]
    with pytest.raises(ValueError, match="python-version"):
        validate_paths(names, "npm")


def test_distribution_does_not_follow_symlinks(tmp_path: Path):
    archive_path = tmp_path / "unsafe.tgz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("package/private")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        archive.addfile(member, io.BytesIO())
    with pytest.raises(ValueError, match="links"):
        validate_archive(archive_path)
