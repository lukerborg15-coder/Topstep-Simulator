from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class FrozenParamsViolation(Exception):
    pass


def _param_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()


def _frozen_path(
    output_dir: Path,
    strategy: str,
    timeframe: str,
    *,
    frozen_params_dir: Path | None = None,
) -> Path:
    if frozen_params_dir is not None:
        return Path(frozen_params_dir) / f"{strategy}_{timeframe}.json"
    return Path(output_dir) / "frozen_params" / f"{strategy}_{timeframe}.json"


def freeze_params(
    output_dir: Path,
    strategy: str,
    timeframe: str,
    params: dict,
    *,
    frozen_params_dir: Path | None = None,
) -> str:
    frozen_hash = _param_hash(params)
    path = _frozen_path(output_dir, strategy, timeframe, frozen_params_dir=frozen_params_dir)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("sha256") == frozen_hash:
            return frozen_hash
        raise FrozenParamsViolation(f"already frozen: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "strategy": strategy,
        "timeframe": timeframe,
        "params": params,
        "sha256": frozen_hash,
        "frozen_at": datetime.now().astimezone().isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return frozen_hash


def verify_frozen_params(
    output_dir: Path,
    strategy: str,
    timeframe: str,
    params: dict,
    *,
    frozen_params_dir: Path | None = None,
) -> str:
    path = _frozen_path(output_dir, strategy, timeframe, frozen_params_dir=frozen_params_dir)
    if not path.exists():
        raise FrozenParamsViolation(f"no frozen file at {path}")

    existing = json.loads(path.read_text())
    frozen_hash = existing.get("sha256")
    current_hash = _param_hash(params)
    if frozen_hash != current_hash:
        raise FrozenParamsViolation(f"hash mismatch: frozen={frozen_hash}, current={current_hash}")
    return current_hash
