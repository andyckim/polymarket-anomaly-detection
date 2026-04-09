"""
AnomalyDetector — scores individual trades against AnomalyConfig thresholds.

Each factor returns a float score ≥ 0.  Scores are summed and compared
against config.scoring.alert_threshold to decide whether to emit an alert.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from config import AnomalyConfig
from polymarket_client import PolymarketClient

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Result dataclasses
# ------------------------------------------------------------------ #

@dataclass
class FactorScore:
    name: str
    score: float
    detail: str


@dataclass
class AnomalyResult:
    trade: dict
    market: dict
    total_score: float
    factors: list[FactorScore]
    severity: str          # "LOW", "MEDIUM", "HIGH"
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        lines = [
            f"[{self.severity}] Anomaly score {self.total_score:.2f}",
            f"  Market : {self.market.get('question', self.market.get('conditionId', '?'))}",
            f"  Trade  : {self.trade.get('usdcSize', '?')} USDC  "
            f"@ {self.trade.get('price', '?')}  side={self.trade.get('side', '?')}",
            f"  Wallet : {self.trade.get('proxyWallet', '?')}",
        ]
        for f in self.factors:
            if f.score > 0:
                lines.append(f"    [{f.score:+.2f}] {f.name}: {f.detail}")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
# Rolling volume tracker (in-memory, per-market)
# ------------------------------------------------------------------ #

class RollingVolumeTracker:
    """Tracks trade timestamps+sizes in a sliding time window per market."""

    def __init__(self, window_minutes: int):
        self.window_seconds = window_minutes * 60
        # condition_id -> list of (timestamp, usdc_size)
        self._buckets: dict[str, list[tuple[float, float]]] = {}

    def record(self, condition_id: str, usdc_size: float):
        bucket = self._buckets.setdefault(condition_id, [])
        bucket.append((time.time(), usdc_size))
        self._prune(condition_id)

    def window_volume(self, condition_id: str) -> float:
        self._prune(condition_id)
        return sum(s for _, s in self._buckets.get(condition_id, []))

    def _prune(self, condition_id: str):
        cutoff = time.time() - self.window_seconds
        bucket = self._buckets.get(condition_id, [])
        self._buckets[condition_id] = [(t, s) for t, s in bucket if t >= cutoff]


# ------------------------------------------------------------------ #
# Main detector
# ------------------------------------------------------------------ #

class AnomalyDetector:
    def __init__(self, config: AnomalyConfig, client: PolymarketClient):
        self.config = config
        self.client = client
        self._volume_tracker = RollingVolumeTracker(
            config.volume.volume_spike_window_minutes
        )
        # Baseline average window volume per market (updated lazily)
        self._baseline_volume: dict[str, float] = {}
        self._baseline_sample_count: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    async def score_trade(
        self,
        trade: dict,
        market: dict,
        pre_trade_price: Optional[float] = None,
    ) -> Optional[AnomalyResult]:
        """
        Score a single trade dict against the config.
        Returns an AnomalyResult if the score meets the alert threshold,
        otherwise returns None.
        """
        cfg = self.config
        usdc_size = float(trade.get("usdcSize") or 0)

        # Skip trades below minimum size
        if usdc_size < cfg.trade_size.min_trade_size_usd:
            return None

        condition_id = trade.get("conditionId", "")
        wallet = trade.get("proxyWallet", "")
        factors: list[FactorScore] = []

        # --- Factor 1: trade size ---
        factors.append(self._score_trade_size(usdc_size))

        # --- Factor 2: account age / history ---
        factors.append(await self._score_account(wallet))

        # --- Factor 3: volume dominance ---
        market_24h_vol = float(market.get("volume24hr") or market.get("volume") or 0)
        factors.append(self._score_volume_dominance(usdc_size, market_24h_vol))

        # --- Factor 4: volume spike ---
        self._volume_tracker.record(condition_id, usdc_size)
        factors.append(self._score_volume_spike(condition_id, usdc_size))

        # --- Factor 5: price impact ---
        post_price = float(trade.get("price") or 0)
        factors.append(self._score_price_impact(pre_trade_price, post_price))

        # --- Factor 6: timing ---
        factors.append(self._score_timing(market, trade))

        # Clamp each factor
        max_f = cfg.scoring.max_single_factor_score
        for f in factors:
            f.score = min(f.score, max_f)

        total = sum(f.score for f in factors)

        # Optional liquidity weighting
        if cfg.scoring.weight_by_liquidity and market_24h_vol > 0:
            liquidity_factor = max(1.0, cfg.volume.thin_market_volume_usd / market_24h_vol)
            total *= min(liquidity_factor, 2.0)   # cap amplification at 2×

        if total < cfg.scoring.alert_threshold:
            return None

        severity = "HIGH" if total >= cfg.scoring.high_severity_threshold else \
                   "MEDIUM" if total >= cfg.scoring.alert_threshold * 1.5 else "LOW"

        return AnomalyResult(
            trade=trade,
            market=market,
            total_score=round(total, 3),
            factors=factors,
            severity=severity,
        )

    # ------------------------------------------------------------------ #
    # Factor scorers
    # ------------------------------------------------------------------ #

    def _score_trade_size(self, usdc_size: float) -> FactorScore:
        cfg = self.config.trade_size
        if usdc_size >= cfg.whale_trade_threshold_usd:
            score = cfg.large_trade_score_bonus + cfg.whale_trade_score_bonus
            detail = f"${usdc_size:,.0f} USDC — whale threshold exceeded"
        elif usdc_size >= cfg.large_trade_threshold_usd:
            score = cfg.large_trade_score_bonus
            detail = f"${usdc_size:,.0f} USDC — large trade threshold exceeded"
        else:
            score = 0.0
            detail = f"${usdc_size:,.0f} USDC — below large threshold"
        return FactorScore("trade_size", score, detail)

    async def _score_account(self, wallet: str) -> FactorScore:
        if not wallet:
            return FactorScore("account", 0.0, "no wallet address")

        cfg = self.config.account
        score = 0.0
        details = []

        try:
            first_ts = await self.client.get_wallet_first_trade_timestamp(wallet)
            if first_ts:
                age_days = (time.time() - first_ts) / 86400
                if age_days <= cfg.new_account_age_days:
                    score += (cfg.new_account_score_multiplier - 1.0)
                    details.append(f"account age {age_days:.0f}d ≤ {cfg.new_account_age_days}d threshold")

            prior_trades = await self.client.get_wallet_trade_count(wallet)
            if prior_trades < cfg.min_prior_trades_for_established:
                score += (cfg.low_history_score_multiplier - 1.0)
                details.append(f"only {prior_trades} prior trades")
        except Exception as e:
            log.warning("account scoring failed for %s: %s", wallet, e)
            details.append("lookup failed")

        return FactorScore("account", score, "; ".join(details) or "established account")

    def _score_volume_dominance(self, usdc_size: float, market_24h_vol: float) -> FactorScore:
        cfg = self.config.volume
        if market_24h_vol <= 0:
            return FactorScore("volume_dominance", 0.0, "no volume data")

        fraction = usdc_size / market_24h_vol
        if fraction >= cfg.trade_volume_fraction_threshold:
            base = fraction / cfg.trade_volume_fraction_threshold  # scales with dominance
            if market_24h_vol < cfg.thin_market_volume_usd:
                base *= cfg.thin_market_score_multiplier
                detail = f"{fraction:.1%} of 24h vol (thin market)"
            else:
                detail = f"{fraction:.1%} of 24h vol"
            return FactorScore("volume_dominance", base, detail)

        return FactorScore("volume_dominance", 0.0, f"{fraction:.1%} of 24h vol — normal")

    def _score_volume_spike(self, condition_id: str, usdc_size: float) -> FactorScore:
        cfg = self.config.volume
        window_vol = self._volume_tracker.window_volume(condition_id)

        # Update rolling baseline
        n = self._baseline_sample_count.get(condition_id, 0)
        baseline = self._baseline_volume.get(condition_id, window_vol)
        # Exponential moving average
        alpha = 0.1
        new_baseline = alpha * window_vol + (1 - alpha) * baseline
        self._baseline_volume[condition_id] = new_baseline
        self._baseline_sample_count[condition_id] = n + 1

        if baseline <= 0 or n < 5:
            return FactorScore("volume_spike", 0.0, "insufficient baseline data")

        ratio = window_vol / baseline
        if ratio >= cfg.volume_spike_multiplier:
            score = cfg.volume_spike_score_bonus * (ratio / cfg.volume_spike_multiplier)
            return FactorScore("volume_spike", score, f"{ratio:.1f}× baseline in rolling window")

        return FactorScore("volume_spike", 0.0, f"{ratio:.1f}× baseline — normal")

    def _score_price_impact(
        self,
        pre_price: Optional[float],
        post_price: float,
    ) -> FactorScore:
        cfg = self.config.price_impact
        if pre_price is None or post_price <= 0:
            return FactorScore("price_impact", 0.0, "no pre-trade price available")

        impact = abs(post_price - pre_price)
        if impact < cfg.min_price_impact:
            return FactorScore("price_impact", 0.0, f"impact {impact:.3f} below threshold")

        score = ((impact - cfg.min_price_impact) / 0.01) * cfg.price_impact_score_per_cent

        details = [f"price moved {impact:.3f}"]
        if post_price >= cfg.high_conviction_price_threshold:
            score += cfg.high_conviction_score_bonus
            details.append(f"price now {post_price:.2f} ≥ conviction threshold")

        return FactorScore("price_impact", score, "; ".join(details))

    def _score_timing(self, market: dict, trade: dict) -> FactorScore:
        cfg = self.config.timing
        hours_left = self.client.hours_until_close(market)

        if hours_left is None:
            return FactorScore("timing", 0.0, "no close time available")

        score = 0.0
        details = []

        if 0 < hours_left <= cfg.late_trade_window_hours:
            score += (cfg.late_trade_score_multiplier - 1.0)
            details.append(f"{hours_left:.1f}h until close — late window")

        # Check if trade happened very close to market open
        # (use market startDate if available)
        start_str = market.get("startDate") or market.get("start_date_iso")
        if start_str:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                hours_since_open = (time.time() - dt.timestamp()) / 3600
                usdc_size = float(trade.get("usdcSize") or 0)
                if (hours_since_open <= cfg.early_trade_window_hours and
                        usdc_size >= self.config.trade_size.large_trade_threshold_usd):
                    score += cfg.early_large_trade_score_bonus
                    details.append(f"{hours_since_open:.1f}h after open — large early trade")
            except ValueError:
                pass

        return FactorScore("timing", score, "; ".join(details) or "normal timing")
