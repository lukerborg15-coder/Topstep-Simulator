from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .verdict import VerdictResult, verdict_summary_dict


_SAFE_STRATEGY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _audit_payload(strategy: str, params_hash: str, verdict: VerdictResult) -> dict[str, Any]:
    summary = verdict_summary_dict(verdict)
    summary["strategy"] = strategy
    return {
        "strategy": strategy,
        "params_hash": params_hash,
        **summary,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sprint": "3",
    }


def write_audit_stamp(
    strategy: str,
    params_hash: str,
    verdict: VerdictResult,
    output_dir: Path,
) -> Path:
    """Write the latest per-strategy audit stamp and append the audit log."""
    if strategy != verdict.strategy:
        raise ValueError(f"strategy mismatch: strategy={strategy!r}, verdict.strategy={verdict.strategy!r}")
    if not _SAFE_STRATEGY_RE.fullmatch(strategy):
        raise ValueError(f"invalid strategy for audit filename: {strategy!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _audit_payload(strategy, params_hash, verdict)

    audit_path = output_dir / f"{strategy}_audit.json"
    tmp_path = audit_path.with_name(f"{audit_path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(audit_path)

    log_path = output_dir / "audit_log.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

    return audit_path


__all__ = ["write_audit_stamp"]
