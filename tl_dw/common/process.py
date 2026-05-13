from __future__ import annotations

import subprocess
from typing import Sequence


def run_command(command: Sequence[str]) -> str:
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        details = stderr or stdout or "Unknown command failure"
        rendered_command = " ".join(command)
        raise RuntimeError(f"Command failed: {rendered_command}\n{details}")
    return proc.stdout.strip()
