from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TradeResult:
    strategy: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry: float
    stop: float
    target: float
    exit: float
    contracts: int
    gross_pnl: float
    commission: float
    net_pnl: float
    r_multiple: float
    exit_reason: str
    bars_held: int
    regime: str = "unknown"
    params: dict[str, Any] = field(default_factory=dict)

__all__ = ["TradeResult"]
