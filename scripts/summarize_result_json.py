"""
Summarize v3.cli result JSON to a readable .txt (drops per-trade arrays).
Usage: python scripts/summarize_result_json.py [path/to/result.json]
Default: output/json/hl2_sma_retrace_atr_5min_result.json next to repo root (use --output-dir parent).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from v3.json_readable import pipeline_result_bundle_to_readable_text  # noqa: E402


def summarize(data: dict) -> str:
    """Backward-compatible alias for ``pipeline_result_bundle_to_readable_text``."""

    return pipeline_result_bundle_to_readable_text(data)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    default_json = repo / "output" / "json" / "hl2_sma_retrace_atr_5min_result.json"
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_json
    if not in_path.is_file():
        print(f"Not found: {in_path}", file=sys.stderr)
        return 1
    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print("Unexpected: root JSON value must be an object", file=sys.stderr)
        return 1
    out_path = in_path.with_name(in_path.stem + "_summary.txt")
    out_path.write_text(summarize(data), encoding="utf-8")
    print(out_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
