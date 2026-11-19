from __future__ import annotations

import atexit
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


class SubprocessOCREngine:
    """Run RapidOCR away from faster-whisper's bundled FFmpeg libraries.

    PyAV and OpenCV ship different FFmpeg builds on macOS. Loading both in one
    process emits duplicate Objective-C class warnings and can crash. A
    persistent worker keeps model startup amortized while isolating the native
    libraries.
    """

    def __init__(self) -> None:
        self._process = subprocess.Popen(
            [sys.executable, "-m", "tl_dw.local_video_to_doc.ocr_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if self._process.stdout is None:
            raise RuntimeError("Unable to read from the OCR worker.")
        ready_line = self._process.stdout.readline()
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            self.close()
            raise RuntimeError(
                f"OCR worker failed to initialize: {ready_line.strip() or 'no response'}"
            ) from exc
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(
                f"OCR worker failed to initialize: {ready.get('error', 'unknown error')}"
            )
        atexit.register(self.close)

    def __call__(self, image_path: str) -> tuple[list[Any], Any]:
        if self._process.poll() is not None:
            raise RuntimeError("OCR worker exited unexpectedly.")
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("OCR worker pipes are unavailable.")
        self._process.stdin.write(
            json.dumps({"image_path": image_path}, ensure_ascii=False) + "\n"
        )
        self._process.stdin.flush()
        response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError("OCR worker exited without returning a result.")
        response = json.loads(response_line)
        if "error" in response:
            raise RuntimeError(f"OCR failed for {image_path}: {response['error']}")
        return response.get("result") or [], response.get("elapsed")

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        if process.stdin is not None:
            try:
                process.stdin.write(json.dumps({"close": True}) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class OllamaOCREngine:
    """Local GLM-OCR served by Ollama (e.g. glm-ocr:q8_0).

    No hosted vision API: the model runs on the local daemon, receives the
    frame as base64 and returns extracted text. Output keeps the pipeline
    tuple shape (bbox=None, text, score) with a nominal score so the
    downstream min_score/min_chars filters behave as configured.
    """

    _PROMPT = (
        "Extract all visible text from this image. Reply with the text lines "
        "only, one per line, preserving the original language. No commentary."
    )

    def __init__(
        self, model: str = "glm-ocr:q8_0", base_url: str | None = None
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("TL_DW_OLLAMA_URL") or "").rstrip(
            "/"
        ) or ("http://localhost:11434")

    def _generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))

    def __call__(self, image_path: str) -> tuple[list[list[Any]], float]:
        started = time.monotonic()
        with open(image_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        response = self._generate(
            {
                "model": self.model,
                "prompt": self._PROMPT,
                "images": [encoded],
                "stream": False,
                "options": {"temperature": 0},
            }
        )
        text = str(response.get("response") or "").strip()
        rows: list[list[Any]] = [
            [None, line.strip(), 0.95] for line in text.splitlines() if line.strip()
        ]
        return rows, round(time.monotonic() - started, 3)

    def close(self) -> None:  # daemon keeps running; nothing to release
        return None


def ollama_glm_ocr_available(
    model_prefix: str = "glm-ocr", base_url: str | None = None
) -> bool:
    """True when the local Ollama daemon exposes a GLM-OCR model."""

    base = (base_url or os.environ.get("TL_DW_OLLAMA_URL") or "").rstrip("/") or (
        "http://localhost:11434"
    )
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False
    for entry in payload.get("models") or []:
        if str(entry.get("name") or "").startswith(model_prefix):
            return True
    return False


def load_ocr_engine() -> "SubprocessOCREngine | OllamaOCREngine":
    """Pick the OCR backend: local Ollama GLM-OCR when available, RapidOCR otherwise.

    TL_DW_OCR_BACKEND=ollama forces Ollama, =rapidocr forces the subprocess
    engine; unset/auto prefers the local GLM-OCR daemon and falls back.
    """

    choice = os.environ.get("TL_DW_OCR_BACKEND", "auto").strip().lower()
    if choice == "rapidocr":
        return SubprocessOCREngine()
    if choice == "ollama" or (choice == "auto" and ollama_glm_ocr_available()):
        return OllamaOCREngine(model=os.environ.get("TL_DW_OCR_MODEL", "glm-ocr:q8_0"))
    return SubprocessOCREngine()
