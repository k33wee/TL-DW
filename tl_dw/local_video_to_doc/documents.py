"""Explicit, bounded, local-only document conversion and embedded raster figures."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from PIL import Image

from .artifacts import (
    PreparationCache,
    artifact_file,
    atomic_bytes,
    atomic_json,
    atomic_text,
    sha256_file,
    signature,
)

TEXT_EXTENSIONS = {".md", ".txt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
OFFICE_EXTENSIONS = {
    ".doc",
    ".docx",
    ".docm",
    ".ppt",
    ".pptx",
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
    ".pdf",
}
DOCUMENT_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | OFFICE_EXTENSIONS
MAX_DOCUMENTS = 20
MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARS = 200_000
MAX_FIGURES = 40
MAX_IMAGE_PIXELS = 20_000_000
DOCUMENT_VERSION = 1


def resolve_documents(inputs: Sequence[Path]) -> list[Path]:
    candidates = []
    for value in inputs:
        source = value.expanduser().resolve()
        if source.is_dir():
            candidates.extend(
                p.resolve()
                for p in sorted(source.iterdir())
                if not p.name.startswith(".")
                and p.is_file()
                and p.suffix.lower() in DOCUMENT_EXTENSIONS
            )
        else:
            candidates.append(source)
    result = sorted(set(candidates))
    if len(result) > MAX_DOCUMENTS:
        raise ValueError(
            f"At most {MAX_DOCUMENTS} explicit context documents are allowed."
        )
    for source in result:
        if not source.is_file() or source.suffix.lower() not in DOCUMENT_EXTENSIONS:
            raise ValueError(f"Unsupported context document: {source}")
        if source.stat().st_size > MAX_DOCUMENT_BYTES:
            raise ValueError(
                f"Context document exceeds {MAX_DOCUMENT_BYTES} bytes: {source.name}"
            )
    if inputs and not result:
        raise ValueError("No supported context documents in the supplied paths.")
    return result


def _validate_container(data: bytes) -> None:
    if not zipfile.is_zipfile(io.BytesIO(data)):
        return
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) > 10_000 or sum(i.file_size for i in infos) > 128 * 1024 * 1024:
            raise ValueError("Document archive exceeds decompression limits")
        seen = set()
        for item in infos:
            name = item.filename.replace("\\", "/")
            if (
                name in seen
                or name.startswith("/")
                or ":" in name
                or ".." in PurePosixPath(name).parts
                or item.flag_bits & 1
                or stat.S_ISLNK(item.external_attr >> 16)
                or item.file_size > 24 * 1024 * 1024
            ):
                raise ValueError(f"Unsafe document archive entry: {item.filename}")
            seen.add(name)


def _figure(data: bytes, root: Path) -> dict:
    if len(data) > 6 * 1024 * 1024:
        raise ValueError(
            "Embedded figure exceeds 6 MiB; supply a smaller document/figure"
        )
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("Embedded figure exceeds the pixel limit")
        image.load()
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba).convert("RGB")
        image.thumbnail((1600, 1600))
        target = io.BytesIO()
        image.save(target, format="PNG")
        normalized = target.getvalue()
    if len(normalized) > 6 * 1024 * 1024:
        raise ValueError("Normalized figure exceeds 6 MiB; reduce its dimensions")
    digest = hashlib.sha256(normalized).hexdigest()
    relative = f"documents/figures/{digest}.png"
    destination = artifact_file(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (root / relative).is_symlink():
        raise ValueError(f"Refusing a symlink context figure: {relative}")
    if not destination.exists() or sha256_file(destination) != digest:
        atomic_bytes(destination, normalized)
    return {"image_path": relative, "sha256": digest}


def prepare_documents(
    inputs: Sequence[Path],
    root: Path,
    *,
    force: bool = False,
    cache: PreparationCache | None = None,
) -> dict:
    sources = resolve_documents(inputs)
    cache = cache if cache is not None else PreparationCache(root, force=force)
    identities = [
        {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sources
    ]
    analysis_path = artifact_file(root, "analysis.json")
    analysis_sha256 = sha256_file(analysis_path) if analysis_path.is_file() else None
    key = signature(
        {
            "version": DOCUMENT_VERSION,
            "sources": identities,
            "analysis_sha256": analysis_sha256,
        }
    )
    if cache.reusable("documents", key):
        return json.loads((root / "documents.json").read_text(encoding="utf-8"))
    cache.begin("documents", key)
    entries, outputs = [], []
    total_chars = 0
    figures_seen = set()
    for index, (source, identity) in enumerate(
        zip(sources, identities, strict=True), 1
    ):
        data = source.read_bytes()
        if (
            len(data) > MAX_DOCUMENT_BYTES
            or hashlib.sha256(data).hexdigest() != identity["sha256"]
        ):
            raise ValueError(
                "Context document changed during conversion; retry with stable inputs"
            )
        suffix = source.suffix.lower()
        figures, warnings = [], []
        if suffix in TEXT_EXTENSIONS:
            markdown = data.decode("utf-8-sig")
        elif suffix in IMAGE_EXTENSIONS:
            markdown = "[Explicit image attachment; see the associated figure.]"
            figures.append(_figure(data, root))
        else:
            _validate_container(data)
            import anydoc

            format_hint = anydoc.format_from_path(source) if suffix == ".csv" else None
            # Never opt in to hosted OCR or a network fallback, even if env API keys exist.
            markdown = anydoc.to_markdown_bytes(data, format_hint, ocr="reject")
            if suffix == ".pdf":
                warnings.append(
                    "PDF text only: embedded/vector diagrams are not rendered. Supply important diagrams as explicit raster attachments."
                )
            else:
                document = anydoc.to_document(data, format_hint)
                for asset in document.assets:
                    if asset.media_type in {
                        "image/png",
                        "image/jpeg",
                        "image/webp",
                        "image/gif",
                    }:
                        figure = _figure(asset.data, root)
                        figure["origin_part"] = asset.origin_part
                        if figure["image_path"] not in {
                            f["image_path"] for f in figures
                        }:
                            figures.append(figure)
                    else:
                        warnings.append(
                            f"Embedded asset not sent as image: {asset.media_type} ({asset.origin_part})"
                        )
        total_chars += len(markdown)
        figures_seen.update(f["image_path"] for f in figures)
        if total_chars > MAX_TEXT_CHARS or len(figures_seen) > MAX_FIGURES:
            raise ValueError(
                f"Context exceeds {MAX_TEXT_CHARS} characters or {MAX_FIGURES} figures. Select a smaller explicit set."
            )
        relative = f"documents/{identity['sha256']}/document.md"
        destination = artifact_file(root, relative)
        atomic_text(destination, markdown)
        outputs.extend(
            [destination, *(artifact_file(root, f["image_path"]) for f in figures)]
        )
        entries.append(
            {
                "id": f"D{index}",
                "name": source.name,
                "source": identity,
                "markdown_path": relative,
                "text_chars": len(markdown),
                "figures": figures,
                "warnings": warnings,
            }
        )
    payload = {
        "schema_version": 1,
        "signature": key,
        "conversion": "local; hosted OCR rejected",
        "analysis_sha256": analysis_sha256,
        "documents": entries,
        "text_chars": total_chars,
        "unique_figures": len(figures_seen),
    }
    atomic_json(root / "documents.json", payload)
    cache.complete("documents", key, [root / "documents.json", *outputs])
    return payload
