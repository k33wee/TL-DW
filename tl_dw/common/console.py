from __future__ import annotations

import os


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def log(message: str) -> None:
    print(message, flush=True)


def vlog(enabled: bool, message: str) -> None:
    if enabled:
        log(f"[verbose] {message}")
