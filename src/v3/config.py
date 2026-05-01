from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

EASTERN_TZ = "America/New_York"
SESSION_START = "09:30"
SESSION_END = "15:00"

# Resolve project root from this file's location:
#   src/v3/config.py -> src/v3 -> src -> <project root>
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

# Data + output dirs are absolute paths anchored at project root, not cwd.
# Override either via env var without touching the code.
DEFAULT_DATA_DIR = Path(os.environ.get("TOPSTEP_PIPELINE_DATA_DIR", str(PROJECT_ROOT / "Data")))
OUTPUT_DIR = Path(os.environ.get("TOPSTEP_PIPELINE_OUTPUT_DIR", str(PROJECT_ROOT / "output")))

TIMEFRAMES = ("1min", "5min", "15min")
STRATEGY_NAMES = (
    "connors_rsi2",
    "ttm_squeeze",
    "orb_ib",
    "orb_volatility_filtered",
    "orb_wick_rejection",
    "session_pivot_rejection",
    "session_pivot_break",
)


@dataclass(frozen=True)
class DateWindow:
    name: str
    start: str
    end: str


@dataclass(frozen=True)
class WalkForwardWindow:
    name: str
    train: DateWindow
    test: DateWindow


@dataclass(frozen=True)
class PipelineWindows:
    in_sample_sanity: DateWindow
    walk_forward: tuple[WalkForwardWindow, ...]
    holdout: DateWindow


WINDOWS = PipelineWindows(
    in_sample_sanity=DateWindow("in_sample_sanity", "2022-09-01", "2024-08-31"),
    walk_forward=(
        WalkForwardWindow(
            "WF1",
            DateWindow("WF1_train", "2022-09-01", "2023-08-31"),
            DateWindow("WF1_test", "2023-09-01", "2023-11-30"),
        ),
        WalkForwardWindow(
            "WF2",
            DateWindow("WF2_train", "2022-09-01", "2023-11-30"),
            DateWindow("WF2_test", "2023-12-01", "2024-02-29"),
        ),
        WalkForwardWindow(
            "WF3",
            DateWindow("WF3_train", "2022-09-01", "2024-02-29"),
            DateWindow("WF3_test", "2024-03-01", "2024-05-31"),
        ),
        WalkForwardWindow(
            "WF4",
            DateWindow("WF4_train", "2022-09-01", "2024-05-31"),
            DateWindow("WF4_test", "2024-06-01", "2024-08-31"),
        ),
    ),
    holdout=DateWindow("holdout", "2024-09-01", "2026-03-18"),
)


@dataclass(frozen=True)
class Instrument:
    symbol: str = "MNQ"
    point_value: float = 2.0
    tick_size: float = 0.25
    commission_round_turn: float = 1.40
    slippage_points_per_side: float = 0.25


MNQ = Instrument()


@dataclass(frozen=True)
class TopStepRules:
    account_size: float = 50_000.0
    profit_target: float = 3_000.0
    max_drawdown: float = 2_000.0
    daily_loss_limit: float = 1_000.0
    min_trading_days: int = 5
    max_micro_contracts: int = 50
    consistency_pct_of_target: float = 0.50


TOPSTEP_50K = TopStepRules()


@dataclass(frozen=True)
class FundedExpressSimRules:
    """Express-funded-style max-loss milestones (approximate vs official wording).

    See Topstep Express Funded Account Rules (maximum loss tied to highest EOD balance):
        https://www.topstep.com/express-funded-account-rules/

    Numeric defaults follow the commonly quoted funded account lock-milestone story
    (trigger at ``SB + max_drawdown + $100``, floor at ``SB + $100`` for Combine-style $50k
    sizing). Topstep wording can change; override fields if your product differs.
    """

    account_size: float = 50_000.0
    max_drawdown: float = 2_000.0
    daily_loss_limit: float = 1_000.0
    lock_trigger_balance: float = 52_100.0
    locked_floor_balance: float = 50_100.0


DEFAULT_FUNDED_EXPRESS_SIM = FundedExpressSimRules()


@dataclass(frozen=True)
class ScoringWeights:
    """
    Weights for strategy scoring during walk-forward optimization.

    Score = topstep_score * topstep_weight + avg_r * avg_r_weight

    Rationale:
    - topstep_score (0-500): Captures Topstep pass (500), speed to pass (up to 30),
      minus penalties for max_dd (5 points per $100) and daily_loss (5 points per $100)
    - avg_r (typically 0.5 to 2.0): Captures consistency (profits per risk unit)

    The 25x multiplier makes avg_r competitive with topstep_score:
    - avg_r of 0.8 -> 20 points (vs topstep_score of 0-500)
    - avg_r of 2.0 -> 50 points

    Tune this if you observe overfitting toward high-variance strategies
    or if you want to prefer consistency over raw Topstep score.
    """

    topstep_weight: float = 1.0
    avg_r_weight: float = 25.0


SCORING_WEIGHTS = ScoringWeights()


@dataclass(frozen=True)
class VerdictThresholds:
    """
    Thresholds for Topstep strategy verdict.

    A strategy is REJECTED if it fails ANY of the reject thresholds.
    A strategy is COMBINE-READY if it passes ALL of the ready thresholds.
    Otherwise it's PROMISING.

    Thresholds are based on TopStep 50k Combine rules:
    - Account: $50,000
    - Target: $3,000 profit
    - Max DD: $2,000
    - Daily loss limit: $1,000

    Reject thresholds are conservative (let bad strategies through to PROMISING).
    Ready thresholds are strict (only best strategies get COMBINE-READY).
    """

    # REJECT if pass rate is below 50%; strategy fails Topstep > 50% of time.
    reject_pass_rate_pct: float = 50.0
    # REJECT if worst-case drawdown is too close to the $2,000 limit.
    reject_max_dd: float = 1_800.0
    # REJECT if daily loss limit is hit often enough to make results unreliable.
    reject_daily_hit_pct: float = 60.0
    # REJECT if average drawdown is high enough to be unsustainable.
    reject_mean_dd: float = 1_200.0

    # COMBINE-READY requires a comfortable pass-rate margin.
    ready_pass_rate_pct: float = 75.0
    # COMBINE-READY requires a worst-case drawdown buffer to the $2,000 limit.
    ready_max_dd: float = 1_400.0
    # COMBINE-READY requires rare daily loss limit hits.
    ready_daily_hit_pct: float = 25.0
    # COMBINE-READY requires sustainable average drawdown.
    ready_mean_dd: float = 800.0


VERDICT_THRESHOLDS = VerdictThresholds()

# Legacy reference range (no longer enforced in `_validate_eval_risk`).
MIN_EVAL_RISK_DOLLARS = 400.0
MAX_EVAL_RISK_DOLLARS = 500.0
DEFAULT_RISK_DOLLARS = 500.0
DEFAULT_MAX_CONTRACTS = TOPSTEP_50K.max_micro_contracts
DEFAULT_MIN_FOLD_SEQ_PASS_RATE_PCT = 40.0
DEFAULT_SKIP_SENSITIVITY = True
DEFAULT_STRICT_WF_GATE = False
