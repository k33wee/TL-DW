from __future__ import annotations

import atexit
import json
import subprocess
import sys
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


def load_ocr_engine() -> SubprocessOCREngine:
    return SubprocessOCREngine()
