"""Content-checked preparation checkpoints. Never trust an incomplete stage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_text(path: Path, text: str) -> None:
    atomic_bytes(path, text.encode("utf-8"))


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tl-dw-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def artifact_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes its directory: {relative}")
    if candidate == root.resolve():
        raise ValueError("An artifact must be a file, not the artifact directory")
    return candidate


@contextmanager
def preparation_lock(root: Path) -> Iterator[None]:
    """OS lock is released even after a killed preparer; the lock file may remain."""
    lock = artifact_file(root, ".preparation.lock")
    with lock.open("a+b") as handle:
        try:
            if os.name == "nt":  # pragma: no cover - Windows
                import msvcrt

                if lock.stat().st_size == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another TL-DW preparation owns {root}") from exc
        try:
            yield
        finally:
            if os.name == "nt":  # pragma: no cover
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class PreparationCache:
    def __init__(self, root: Path, *, force: bool = False):
        self.root = root.resolve()
        self.path = artifact_file(root, "preparation.json")
        self.force = force
        self.executed: set[str] = set()
        self.reused: set[str] = set()
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            self.stages = (
                value.get("stages", {})
                if isinstance(value, dict) and value.get("schema_version") == 1
                else {}
            )
            if not isinstance(self.stages, dict):
                self.stages = {}
        except (OSError, ValueError, KeyError, TypeError):
            self.stages = {}

    def reusable(self, stage: str, key: str) -> bool:
        entry = self.stages.get(stage, {})
        if (
            self.force
            or not isinstance(entry, dict)
            or entry.get("status") != "complete"
            or entry.get("signature") != key
        ):
            return False
        outputs = entry.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            return False
        try:
            valid = all(
                artifact_file(self.root, name).is_file()
                and sha256_file(artifact_file(self.root, name)) == digest
                for name, digest in outputs.items()
            )
            if valid:
                self.reused.add(stage)
            return valid
        except (OSError, ValueError):
            return False

    def begin(self, stage: str, key: str) -> None:
        self.stages[stage] = {"status": "running", "signature": key}
        self._save()

    def complete(self, stage: str, key: str, outputs: list[Path]) -> None:
        self.executed.add(stage)
        self.stages[stage] = {
            "status": "complete",
            "signature": key,
            "outputs": {
                str(p.resolve().relative_to(self.root)): sha256_file(p) for p in outputs
            },
        }
        self._save()

    def _save(self) -> None:
        atomic_json(self.path, {"schema_version": 1, "stages": self.stages})
