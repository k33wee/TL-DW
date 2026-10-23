"""Local media facts and conservative audio selection; never auto-mix distinct tracks."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .artifacts import atomic_json

MAX_AUDIO_TRACKS = 8
SILENCE_PEAK_DB = -80.0


def require_media_tools() -> None:
    for name in ("ffprobe", "ffmpeg"):
        if shutil.which(name) is None:
            raise RuntimeError(
                f"Missing {name} on PATH; install FFmpeg before preparing media."
            )


def media_preflight(
    video: Path, metadata: dict[str, Any], output_root: Path, args
) -> dict:
    streams = metadata.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    visual = [s for s in streams if s.get("codec_type") == "video"]
    duration = float(metadata.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or duration <= 0 or not visual:
        raise ValueError(
            "Recording must contain video and have a finite positive duration."
        )
    if not audio:
        raise ValueError("Recording has no audio stream to transcribe.")
    # Mono PCM plus chunks and a conservative allowance for low-rate frame scanning.
    scan_frames = math.ceil(duration / args.frame_sample_sec) if args.use_frames else 0
    estimated = int(
        duration * 16000 * 2 * 2 + scan_frames * 512 * 1024 + 8 * 1024 * 1024
    )
    free = shutil.disk_usage(output_root).free
    result = {
        "source": str(video),
        "duration_seconds": duration,
        "source_bytes": video.stat().st_size,
        "video": [
            {
                k: s.get(k)
                for k in ("index", "codec_name", "width", "height", "avg_frame_rate")
            }
            for s in visual
        ],
        "audio": [
            {k: s.get(k) for k in ("index", "codec_name", "channels", "sample_rate")}
            for s in audio
        ],
        "estimated_temporary_bytes": estimated,
        "free_bytes": free,
        "space_estimate_scope": "heuristic audio/chunk/frame allowance; excludes model downloads",
        "compression": "Not applied. Local extraction preserves screen text and timestamps.",
    }
    atomic_json(output_root / "preflight.json", result)
    if free < estimated:
        raise RuntimeError(
            f"Insufficient disk space: estimated {estimated} bytes, available {free}. Choose another output volume."
        )
    return result


def _ffmpeg_output(args: list[str]) -> tuple[str, str]:
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", *args], capture_output=True, text=True
    )
    if proc.returncode:
        raise RuntimeError(f"Audio diagnostics failed: {proc.stderr[-2000:]}")
    return proc.stdout, proc.stderr


def diagnose_track(video: Path, stream: dict, duration: float, scope: str) -> dict:
    index = stream["index"]
    # Packet hash covers the full track even in sampled volume mode. Codec/layout are
    # also compared: equal hashes alone do not establish equivalent decoding.
    stdout, _ = _ffmpeg_output(
        [
            "-v",
            "error",
            "-i",
            str(video),
            "-map",
            f"0:{index}",
            "-c:a",
            "copy",
            "-f",
            "hash",
            "-hash",
            "sha256",
            "-",
        ]
    )
    match = re.search(r"SHA256=([a-fA-F0-9]{64})", stdout)
    if not match:
        raise RuntimeError("FFmpeg did not return an audio payload hash")
    intervals = (
        [(0.0, None)]
        if scope == "full"
        else [
            (point, min(5.0, duration - point))
            for point in sorted({0.0, duration / 2, max(0.0, duration - 5)})
        ]
    )
    peaks = []
    for start, length in intervals:
        timing = ["-ss", str(start)] if start else []
        _, stderr = _ffmpeg_output(
            [
                *timing,
                "-i",
                str(video),
                *(["-t", str(length)] if length else []),
                "-map",
                f"0:{index}",
                "-af",
                "volumedetect",
                "-f",
                "null",
                "-",
            ]
        )
        values = re.findall(r"max_volume:\s*(-?inf|[-\d.]+) dB", stderr)
        if not values:
            raise RuntimeError("FFmpeg did not return an audio peak volume")
        peaks.append(float(values[-1]))
    peak = max(peaks)
    return {
        "index": index,
        "codec_name": stream.get("codec_name"),
        "channels": stream.get("channels"),
        "sample_rate": stream.get("sample_rate"),
        **{
            key: stream.get(key)
            for key in (
                "profile",
                "channel_layout",
                "time_base",
                "start_time",
                "duration",
            )
        },
        "payload_sha256": match[1].lower(),
        "volume_scope": scope,
        "sample_intervals": intervals if scope == "sample" else None,
        "max_volume_db": peak if math.isfinite(peak) else None,
        "silent_full_track": scope == "full" and peak <= SILENCE_PEAK_DB,
        "quiet_sample_only": scope == "sample" and peak <= SILENCE_PEAK_DB,
    }


def select_audio(
    video: Path,
    metadata: dict,
    *,
    stream: int | None = None,
    mix: list[int] | None = None,
    scope: str = "full",
) -> dict:
    audio = [s for s in metadata.get("streams", []) if s.get("codec_type") == "audio"]
    available = {s["index"] for s in audio}
    requested = [stream] if stream is not None else mix
    if requested is not None:
        if (
            not requested
            or len(requested) != len(set(requested))
            or not set(requested) <= available
        ):
            raise ValueError(
                f"Select distinct audio stream indices from {sorted(available)}."
            )
        if len(requested) > MAX_AUDIO_TRACKS:
            raise ValueError(f"At most {MAX_AUDIO_TRACKS} tracks can be mixed.")
        return {
            "mode": "explicit",
            "selected_streams": requested,
            "tracks": [],
            "diagnostic_scope": "not_run",
        }
    if len(audio) == 1:
        return {
            "mode": "single",
            "selected_streams": [audio[0]["index"]],
            "tracks": [],
            "diagnostic_scope": "not_needed",
        }
    if len(audio) > MAX_AUDIO_TRACKS:
        raise ValueError(
            f"More than {MAX_AUDIO_TRACKS} audio tracks; choose --audio-stream or --audio-mix explicitly."
        )
    tracks = [
        diagnose_track(video, s, float(metadata["format"]["duration"]), scope)
        for s in audio
    ]
    unique: dict[tuple, int] = {}
    candidates = []
    for track in tracks:
        if track["silent_full_track"]:
            continue
        key = tuple(
            track.get(k)
            for k in (
                "payload_sha256",
                "codec_name",
                "channels",
                "sample_rate",
                "profile",
                "channel_layout",
                "time_base",
                "start_time",
                "duration",
            )
        )
        if key in unique:
            track["duplicate_of"] = unique[key]
        else:
            unique[key] = track["index"]
            candidates.append(track["index"])
    return {
        "mode": "automatic" if len(candidates) == 1 else "ambiguous",
        "selected_streams": candidates if len(candidates) == 1 else [],
        "candidate_streams": candidates,
        "tracks": tracks,
        "diagnostic_scope": scope,
        "silence_peak_threshold_db": SILENCE_PEAK_DB,
    }


def require_audio_selection(result: dict) -> list[int]:
    if result["selected_streams"]:
        return result["selected_streams"]
    candidates = result.get("candidate_streams", [])
    raise ValueError(
        f"Audio selection is ambiguous ({candidates or 'no non-silent track'}). "
        "Inspect audio-selection.json and choose --audio-stream INDEX or "
        "--audio-mix INDEX INDEX ... explicitly. Sampled quietness is not full-track silence."
    )
