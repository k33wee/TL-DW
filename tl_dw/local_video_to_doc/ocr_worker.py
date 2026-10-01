from __future__ import annotations

import json
import sys
from typing import Any


def _serializable_result(result: list[Any] | None) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for item in result or []:
        try:
            _, text, score = item
        except (TypeError, ValueError):
            continue
        rows.append([None, str(text or ""), float(score)])
    return rows


def main() -> int:
    try:
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR()
    except Exception as exc:  # pragma: no cover - dependency/runtime specific
        print(json.dumps({"ready": False, "error": str(exc)}), flush=True)
        return 1

    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("close"):
                return 0
            image_path = request["image_path"]
            result, elapsed = engine(image_path)
            print(
                json.dumps(
                    {
                        "result": _serializable_result(result),
                        "elapsed": elapsed,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
