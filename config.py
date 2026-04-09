"""
AnomalyConfig — all tunable thresholds for the Polymarket anomaly detector.
Edit the values here to change detection sensitivity without touching bot logic.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AccountConfig:
    """Factors related to the trader's account age and history."""

    # Flag trades from wallets whose first-seen trade is within this many days
    new_account_age_days: int = 30

    # Score multiplier applied when a wallet is considered "new"
    # e.g. 1.5 = 50% boost to the anomaly score
    new_account_score_multiplier: float = 1.5

    # Minimum number of prior trades a wallet must have to be considered established
    min_prior_trades_for_established: int = 10

    # Score multiplier when a wallet has fewer trades than the minimum above
    low_history_score_multiplier: float = 1.3


@dataclass
class VolumeConfig:
    """Factors related to market-level and wallet-level trading volume."""

    # A single trade is anomalous if it represents >= this fraction of the
    # market's 24h volume (0.10 = 10%)
    trade_volume_fraction_threshold: float = 0.10

    # Absolute 24h market volume (in USDC) below which a market is considered
    # "thin" — large trades in thin markets score higher
    thin_market_volume_usd: float = 5_000.0

    # Score multiplier for trades that dominate a thin market's volume
    thin_market_score_multiplier: float = 2.0

    # Rolling window (minutes) used when calculating recent volume spikes
    volume_spike_window_minutes: int = 60

    # A spike is detected when current-window volume exceeds the baseline
    # rolling average by this multiple (e.g. 3.0 = 3× the average)
    volume_spike_multiplier: float = 3.0

    # Score added when a volume spike is detected
    volume_spike_score_bonus: float = 1.0


@dataclass
class TradeSizeConfig:
    """Factors related to the absolute and relative size of a single trade."""

    # Minimum trade size (USDC) to even consider for anomaly scoring
    min_trade_size_usd: float = 100.0

    # A trade is "large" if it exceeds this absolute USDC threshold
    large_trade_threshold_usd: float = 10_000.0

    # Score bonus applied to trades above large_trade_threshold_usd
    large_trade_score_bonus: float = 1.0

    # A trade is "whale" if it exceeds this absolute USDC threshold
    whale_trade_threshold_usd: float = 50_000.0

    # Score bonus applied to whale trades (stacks with large_trade_score_bonus)
    whale_trade_score_bonus: float = 2.0


@dataclass
class PriceImpactConfig:
    """Factors related to how much a trade moves the market price."""

    # Minimum price move (absolute, 0–1 scale) to flag as impactful
    # e.g. 0.05 = a move from 0.50 to 0.55 or higher
    min_price_impact: float = 0.05

    # Score bonus per 0.01 of price impact above the minimum
    price_impact_score_per_cent: float = 0.2

    # If a trade pushes price above this threshold (e.g. 0.85 = 85¢),
    # treat it as a potential "last-minute" conviction trade
    high_conviction_price_threshold: float = 0.85

    # Score bonus for trades that push price above high_conviction_price_threshold
    high_conviction_score_bonus: float = 1.5


@dataclass
class TimingConfig:
    """Factors related to *when* a trade occurs relative to market lifecycle."""

    # Hours before market close within which trades are considered "late"
    late_trade_window_hours: float = 24.0

    # Score multiplier applied to trades in the late window
    late_trade_score_multiplier: float = 1.4

    # Hours after market open within which trades are considered "early"
    # (very early large trades can signal informed trading)
    early_trade_window_hours: float = 1.0

    # Score bonus for anomalously large trades placed in the early window
    early_large_trade_score_bonus: float = 0.8


@dataclass
class ScoringConfig:
    """Controls how individual factor scores are combined and thresholded."""

    # Minimum total anomaly score required to emit an alert (tune this first)
    alert_threshold: float = 3.0

    # Score at which an alert is escalated to HIGH severity
    high_severity_threshold: float = 6.0

    # Whether to weight scores by market liquidity
    # (low-liquidity markets amplify the same raw score)
    weight_by_liquidity: bool = True

    # Maximum score any single factor can contribute (prevents one signal
    # from dominating everything else)
    max_single_factor_score: float = 3.0


@dataclass
class AnomalyConfig:
    """
    Top-level configuration object.  Pass this to AnomalyDetector.

    Example — tighten sensitivity across the board:
        cfg = AnomalyConfig()
        cfg.scoring.alert_threshold = 2.0
        cfg.trade_size.large_trade_threshold_usd = 5_000.0
    """

    account: AccountConfig = field(default_factory=AccountConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    trade_size: TradeSizeConfig = field(default_factory=TradeSizeConfig)
    price_impact: PriceImpactConfig = field(default_factory=PriceImpactConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    # Markets to watch — empty list means watch ALL active markets
    watched_markets: list = field(default_factory=list)

    # Markets to explicitly ignore (by condition_id or slug)
    ignored_markets: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== AnomalyConfig ===",
            f"  Alert threshold:          {self.scoring.alert_threshold}",
            f"  High-severity threshold:  {self.scoring.high_severity_threshold}",
            f"  Min trade size (USDC):    {self.trade_size.min_trade_size_usd}",
            f"  Large trade (USDC):       {self.trade_size.large_trade_threshold_usd}",
            f"  Whale trade (USDC):       {self.trade_size.whale_trade_threshold_usd}",
            f"  New account window (days):{self.account.new_account_age_days}",
            f"  Volume spike multiplier:  {self.volume.volume_spike_multiplier}x",
            f"  Late-trade window (hrs):  {self.timing.late_trade_window_hours}",
            f"  Watched markets:          {self.watched_markets or 'ALL'}",
        ]
        return "\n".join(lines)
