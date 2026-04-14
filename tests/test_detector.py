"""
Unit tests for detector.py — RollingVolumeTracker, factor scorers, and AnomalyDetector.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import AnomalyConfig
from detector import AnomalyDetector, AnomalyResult, FactorScore, RollingVolumeTracker


def run(coro):
    return asyncio.run(coro)


def make_test_cfg() -> AnomalyConfig:
    """
    Return a config with stable, pinned values for unit tests.
    These are independent of the production defaults so that tuning
    config.py does not break factor-level tests.
    """
    cfg = AnomalyConfig()
    # trade size
    cfg.trade_size.min_trade_size_usd          = 100.0
    cfg.trade_size.large_trade_threshold_usd   = 10_000.0
    cfg.trade_size.large_trade_score_bonus     = 1.0
    cfg.trade_size.whale_trade_threshold_usd   = 50_000.0
    cfg.trade_size.whale_trade_score_bonus     = 2.0
    # volume
    cfg.volume.trade_volume_fraction_threshold = 0.10
    cfg.volume.thin_market_volume_usd          = 5_000.0
    cfg.volume.thin_market_score_multiplier    = 2.0
    cfg.volume.volume_spike_multiplier         = 3.0
    cfg.volume.volume_spike_score_bonus        = 1.0
    # price impact
    cfg.price_impact.min_price_impact                = 0.05
    cfg.price_impact.price_impact_score_per_cent     = 0.2
    cfg.price_impact.high_conviction_price_threshold = 0.85
    cfg.price_impact.high_conviction_score_bonus     = 1.5
    # timing
    cfg.timing.late_trade_score_multiplier    = 1.4
    cfg.timing.early_large_trade_score_bonus  = 0.8
    # account
    cfg.account.new_account_age_days               = 30
    cfg.account.new_account_score_multiplier       = 1.5
    cfg.account.min_prior_trades_for_established   = 10
    cfg.account.low_history_score_multiplier       = 1.3
    # scoring
    cfg.scoring.alert_threshold          = 3.0
    cfg.scoring.high_severity_threshold  = 6.0
    cfg.scoring.max_single_factor_score  = 3.0
    cfg.scoring.weight_by_liquidity      = True
    return cfg


def make_detector(cfg=None):
    cfg = cfg if cfg is not None else make_test_cfg()
    client = MagicMock()
    client.get_wallet_first_trade_timestamp = AsyncMock(return_value=None)
    client.get_wallet_trade_count = AsyncMock(return_value=100)
    client.hours_until_close = MagicMock(return_value=None)
    return AnomalyDetector(cfg, client), client


# ---------------------------------------------------------------------------
# RollingVolumeTracker
# ---------------------------------------------------------------------------

class TestRollingVolumeTracker:
    def test_unknown_market_returns_zero(self):
        tracker = RollingVolumeTracker(60)
        assert tracker.window_volume("unknown") == 0.0

    def test_records_and_sums_within_window(self):
        tracker = RollingVolumeTracker(60)
        tracker.record("m", 100.0)
        tracker.record("m", 250.0)
        assert tracker.window_volume("m") == 350.0

    def test_old_entries_pruned(self):
        tracker = RollingVolumeTracker(60)  # 60-minute window = 3600 s
        tracker._buckets["m"] = [(time.time() - 3601, 999.0)]
        assert tracker.window_volume("m") == 0.0

    def test_recent_entries_kept(self):
        tracker = RollingVolumeTracker(60)
        tracker._buckets["m"] = [(time.time() - 100, 500.0)]  # 100 s ago, within 3600 s
        assert tracker.window_volume("m") == 500.0

    def test_prune_boundary_exact_cutoff_kept(self):
        """Entry timestamped exactly at the cutoff edge is kept (t >= cutoff)."""
        from unittest.mock import patch as _patch
        with _patch("detector.time") as mock_time:
            mock_time.time.return_value = 1000.0
            tracker = RollingVolumeTracker(1)  # 1-minute = 60 s window
            tracker._buckets["m"] = [
                (940.0, 100.0),   # 1000 - 60 = 940 → exactly at cutoff, kept
                (939.9, 200.0),   # just outside → pruned
            ]
            assert tracker.window_volume("m") == 100.0

    def test_multiple_markets_isolated(self):
        tracker = RollingVolumeTracker(60)
        tracker.record("a", 100.0)
        tracker.record("b", 999.0)
        assert tracker.window_volume("a") == 100.0
        assert tracker.window_volume("b") == 999.0

    def test_record_triggers_prune(self):
        """Stale entries are removed when a new record is added."""
        tracker = RollingVolumeTracker(1)
        tracker._buckets["m"] = [(time.time() - 120, 999.0)]
        tracker.record("m", 50.0)
        assert tracker.window_volume("m") == 50.0


# ---------------------------------------------------------------------------
# _score_trade_size
# ---------------------------------------------------------------------------

class TestScoreTradeSize:
    def test_below_large_threshold_zero_score(self):
        det, _ = make_detector()
        fs = det._score_trade_size(5_000.0)
        assert fs.score == 0.0

    def test_exactly_at_large_threshold(self):
        det, _ = make_detector()
        fs = det._score_trade_size(10_000.0)
        assert fs.score == pytest.approx(det.config.trade_size.large_trade_score_bonus)

    def test_above_large_below_whale(self):
        det, _ = make_detector()
        fs = det._score_trade_size(25_000.0)
        assert fs.score == pytest.approx(det.config.trade_size.large_trade_score_bonus)

    def test_exactly_at_whale_threshold(self):
        det, _ = make_detector()
        fs = det._score_trade_size(50_000.0)
        expected = (det.config.trade_size.large_trade_score_bonus +
                    det.config.trade_size.whale_trade_score_bonus)
        assert fs.score == pytest.approx(expected)

    def test_above_whale_threshold(self):
        det, _ = make_detector()
        fs = det._score_trade_size(200_000.0)
        expected = (det.config.trade_size.large_trade_score_bonus +
                    det.config.trade_size.whale_trade_score_bonus)
        assert fs.score == pytest.approx(expected)

    def test_factor_name(self):
        det, _ = make_detector()
        assert det._score_trade_size(1_000.0).name == "trade_size"


# ---------------------------------------------------------------------------
# _score_volume_dominance
# ---------------------------------------------------------------------------

class TestScoreVolumeDominance:
    def test_zero_market_volume(self):
        det, _ = make_detector()
        fs = det._score_volume_dominance(1_000.0, 0.0)
        assert fs.score == 0.0
        assert "no volume data" in fs.detail

    def test_negative_market_volume(self):
        det, _ = make_detector()
        fs = det._score_volume_dominance(1_000.0, -500.0)
        assert fs.score == 0.0

    def test_fraction_below_threshold(self):
        det, _ = make_detector()
        # fraction = 1000/20000 = 5%, threshold = 10%
        fs = det._score_volume_dominance(1_000.0, 20_000.0)
        assert fs.score == 0.0

    def test_fraction_exactly_at_threshold(self):
        det, _ = make_detector()
        # fraction = 0.10 == threshold → base = 1.0
        fs = det._score_volume_dominance(1_000.0, 10_000.0)
        assert fs.score == pytest.approx(1.0)

    def test_thick_market_no_multiplier(self):
        det, _ = make_detector()
        # fraction = 2000/10000 = 20%, threshold = 10%, base = 2.0
        # market 10k >= thin_market threshold 5k → no thin multiplier
        fs = det._score_volume_dominance(2_000.0, 10_000.0)
        assert fs.score == pytest.approx(2.0)
        assert "thin" not in fs.detail

    def test_thin_market_multiplier_applied(self):
        det, _ = make_detector()
        # fraction = 400/2000 = 20%, base = 2.0, thin_market_multiplier = 2.0
        # 2000 < thin_market_volume_usd (5000)
        fs = det._score_volume_dominance(400.0, 2_000.0)
        expected = (400.0 / 2_000.0) / 0.10 * 2.0
        assert fs.score == pytest.approx(expected)
        assert "thin" in fs.detail

    def test_market_at_exact_thin_boundary_is_not_thin(self):
        """Market with volume == thin_market_volume_usd is NOT thin (strict <)."""
        det, _ = make_detector()
        fs = det._score_volume_dominance(500.0, 5_000.0)
        assert "thin" not in fs.detail


# ---------------------------------------------------------------------------
# _score_volume_spike
# ---------------------------------------------------------------------------

class TestScoreVolumeSpike:
    def _prime_baseline(self, det, condition_id, baseline_vol, n=5):
        det._baseline_sample_count[condition_id] = n
        det._baseline_volume[condition_id] = baseline_vol

    def _set_window_vol(self, det, condition_id, vol):
        det._volume_tracker._buckets[condition_id] = [(time.time(), vol)]

    def test_returns_zero_when_n_less_than_5(self):
        det, _ = make_detector()
        self._prime_baseline(det, "m", 100.0, n=4)
        self._set_window_vol(det, "m", 500.0)
        fs = det._score_volume_spike("m", 500.0)
        assert fs.score == 0.0
        assert "insufficient" in fs.detail

    def test_returns_zero_when_baseline_is_zero(self):
        det, _ = make_detector()
        self._prime_baseline(det, "m", 0.0, n=5)
        self._set_window_vol(det, "m", 100.0)
        fs = det._score_volume_spike("m", 100.0)
        assert fs.score == 0.0

    def test_no_spike_below_multiplier(self):
        det, _ = make_detector()
        self._prime_baseline(det, "m", 100.0, n=5)
        self._set_window_vol(det, "m", 200.0)  # ratio = 2.0 < 3.0 multiplier
        fs = det._score_volume_spike("m", 200.0)
        assert fs.score == 0.0
        assert "normal" in fs.detail

    def test_spike_exactly_at_multiplier(self):
        det, _ = make_detector()
        self._prime_baseline(det, "m", 100.0, n=5)
        self._set_window_vol(det, "m", 300.0)  # ratio = 3.0 == multiplier
        fs = det._score_volume_spike("m", 300.0)
        # score = 1.0 * (3.0 / 3.0) = 1.0
        assert fs.score == pytest.approx(1.0)

    def test_spike_scales_with_ratio(self):
        det, _ = make_detector()
        self._prime_baseline(det, "m", 100.0, n=5)
        self._set_window_vol(det, "m", 400.0)  # ratio = 4.0
        fs = det._score_volume_spike("m", 400.0)
        expected = 1.0 * (4.0 / 3.0)
        assert fs.score == pytest.approx(expected)

    def test_ema_baseline_updated(self):
        det, _ = make_detector()
        self._prime_baseline(det, "m", 100.0, n=5)
        self._set_window_vol(det, "m", 200.0)
        det._score_volume_spike("m", 200.0)
        # new_baseline = 0.1 * 200 + 0.9 * 100 = 110
        assert det._baseline_volume["m"] == pytest.approx(110.0)
        assert det._baseline_sample_count["m"] == 6

    def test_new_market_initializes_baseline(self):
        det, _ = make_detector()
        self._set_window_vol(det, "m", 50.0)
        fs = det._score_volume_spike("m", 50.0)
        assert fs.score == 0.0  # n=0 < 5
        assert det._baseline_sample_count["m"] == 1


# ---------------------------------------------------------------------------
# _score_price_impact
# ---------------------------------------------------------------------------

class TestScorePriceImpact:
    def test_no_pre_price_returns_zero(self):
        det, _ = make_detector()
        fs = det._score_price_impact(None, 0.6)
        assert fs.score == 0.0

    def test_zero_post_price_returns_zero(self):
        det, _ = make_detector()
        fs = det._score_price_impact(0.5, 0.0)
        assert fs.score == 0.0

    def test_negative_post_price_returns_zero(self):
        det, _ = make_detector()
        fs = det._score_price_impact(0.5, -0.1)
        assert fs.score == 0.0

    def test_impact_below_min_returns_zero(self):
        det, _ = make_detector()
        # min_price_impact = 0.05, impact = 0.04
        fs = det._score_price_impact(0.50, 0.54)
        assert fs.score == 0.0

    def test_impact_exactly_at_min_gives_zero_score(self):
        det, _ = make_detector()
        # impact = 0.05, score = (0.05 - 0.05) / 0.01 * 0.2 = 0.0
        fs = det._score_price_impact(0.50, 0.55)
        assert fs.score == pytest.approx(0.0)

    def test_impact_above_min_scores_correctly(self):
        det, _ = make_detector()
        # impact = 0.10, score = (0.10 - 0.05) / 0.01 * 0.2 = 5 * 0.2 = 1.0
        fs = det._score_price_impact(0.50, 0.60)
        assert fs.score == pytest.approx(1.0)

    def test_high_conviction_bonus_added(self):
        det, _ = make_detector()
        # post_price = 0.90 >= 0.85 conviction threshold
        fs = det._score_price_impact(0.80, 0.90)
        base = (0.10 - 0.05) / 0.01 * 0.2  # 1.0
        assert fs.score == pytest.approx(base + 1.5)

    def test_no_conviction_bonus_below_threshold(self):
        det, _ = make_detector()
        # post_price = 0.84 < 0.85
        fs = det._score_price_impact(0.74, 0.84)
        assert fs.score == pytest.approx((0.10 - 0.05) / 0.01 * 0.2)

    def test_price_decrease_uses_absolute_value(self):
        det, _ = make_detector()
        # Price falls 0.10 — impact is same as a 0.10 rise
        fs = det._score_price_impact(0.60, 0.50)
        assert fs.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _score_timing
# ---------------------------------------------------------------------------

class TestScoreTiming:
    def test_no_close_time_returns_zero(self):
        det, client = make_detector()
        client.hours_until_close.return_value = None
        fs = det._score_timing({}, {})
        assert fs.score == 0.0

    def test_market_already_closed(self):
        det, client = make_detector()
        client.hours_until_close.return_value = -1.0
        fs = det._score_timing({}, {})
        assert fs.score == 0.0

    def test_outside_late_window_no_bonus(self):
        det, client = make_detector()
        client.hours_until_close.return_value = 48.0  # > 24 h window
        fs = det._score_timing({}, {})
        assert fs.score == 0.0

    def test_inside_late_window_adds_bonus(self):
        det, client = make_detector()
        client.hours_until_close.return_value = 12.0
        fs = det._score_timing({}, {})
        # late_trade_score_multiplier = 1.4 → bonus = 1.4 - 1.0 = 0.4
        assert fs.score == pytest.approx(0.4)
        assert "late" in fs.detail

    def test_exactly_at_late_window_boundary(self):
        det, client = make_detector()
        client.hours_until_close.return_value = 24.0  # == late_trade_window_hours
        fs = det._score_timing({}, {})
        assert fs.score == pytest.approx(0.4)

    def test_early_large_trade_bonus(self):
        from datetime import datetime, timezone, timedelta
        det, client = make_detector()
        client.hours_until_close.return_value = 200.0
        start = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        market = {"startDate": start}
        trade = {"usdcSize": "15000"}  # >= large_trade_threshold_usd (10 000)
        fs = det._score_timing(market, trade)
        assert fs.score == pytest.approx(0.8)  # early_large_trade_score_bonus

    def test_early_small_trade_no_bonus(self):
        from datetime import datetime, timezone, timedelta
        det, client = make_detector()
        client.hours_until_close.return_value = 200.0
        start = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        market = {"startDate": start}
        trade = {"usdcSize": "500"}  # below large threshold
        fs = det._score_timing(market, trade)
        assert fs.score == 0.0

    def test_trade_after_early_window_no_bonus(self):
        from datetime import datetime, timezone, timedelta
        det, client = make_detector()
        client.hours_until_close.return_value = 200.0
        start = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        market = {"startDate": start}
        trade = {"usdcSize": "50000"}
        fs = det._score_timing(market, trade)
        assert fs.score == 0.0

    def test_start_date_iso_fallback_key(self):
        from datetime import datetime, timezone, timedelta
        det, client = make_detector()
        client.hours_until_close.return_value = 200.0
        start = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        market = {"start_date_iso": start}  # alternate key
        trade = {"usdcSize": "15000"}
        fs = det._score_timing(market, trade)
        assert fs.score == pytest.approx(0.8)

    def test_invalid_start_date_does_not_crash(self):
        det, client = make_detector()
        client.hours_until_close.return_value = 200.0
        fs = det._score_timing({"startDate": "not-a-date"}, {"usdcSize": "20000"})
        assert fs.score == 0.0

    def test_both_late_and_early_bonuses_stack(self):
        from datetime import datetime, timezone, timedelta
        det, client = make_detector()
        client.hours_until_close.return_value = 12.0  # late
        start = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        market = {"startDate": start}
        trade = {"usdcSize": "15000"}
        fs = det._score_timing(market, trade)
        assert fs.score == pytest.approx(0.4 + 0.8)  # 1.2


# ---------------------------------------------------------------------------
# _score_account
# ---------------------------------------------------------------------------

class TestScoreAccount:
    def test_empty_wallet_returns_zero(self):
        det, _ = make_detector()
        fs = run(det._score_account(""))
        assert fs.score == 0.0
        assert "no wallet" in fs.detail

    def test_established_old_account_no_bonus(self):
        det, client = make_detector()
        client.get_wallet_first_trade_timestamp = AsyncMock(
            return_value=int(time.time()) - 90 * 86400)
        client.get_wallet_trade_count = AsyncMock(return_value=50)
        fs = run(det._score_account("0xabc"))
        assert fs.score == 0.0

    def test_new_account_age_bonus(self):
        det, client = make_detector()
        client.get_wallet_first_trade_timestamp = AsyncMock(
            return_value=int(time.time()) - 5 * 86400)  # 5 days
        client.get_wallet_trade_count = AsyncMock(return_value=50)
        fs = run(det._score_account("0xabc"))
        # new_account_score_multiplier = 1.5 → +0.5
        assert fs.score == pytest.approx(0.5)

    def test_low_trade_history_bonus(self):
        det, client = make_detector()
        client.get_wallet_first_trade_timestamp = AsyncMock(
            return_value=int(time.time()) - 90 * 86400)
        client.get_wallet_trade_count = AsyncMock(return_value=3)  # < 10
        fs = run(det._score_account("0xabc"))
        # low_history_score_multiplier = 1.3 → +0.3
        assert fs.score == pytest.approx(0.3)

    def test_both_new_and_low_history_stack(self):
        det, client = make_detector()
        client.get_wallet_first_trade_timestamp = AsyncMock(
            return_value=int(time.time()) - 5 * 86400)
        client.get_wallet_trade_count = AsyncMock(return_value=3)
        fs = run(det._score_account("0xabc"))
        assert fs.score == pytest.approx(0.8)  # 0.5 + 0.3

    def test_trade_count_exactly_at_minimum_no_penalty(self):
        """Exactly min_prior_trades_for_established is NOT penalised (< check)."""
        det, client = make_detector()
        client.get_wallet_first_trade_timestamp = AsyncMock(return_value=None)
        client.get_wallet_trade_count = AsyncMock(return_value=10)  # == minimum
        fs = run(det._score_account("0xabc"))
        assert fs.score == 0.0

    def test_no_first_ts_no_age_penalty(self):
        det, client = make_detector()
        client.get_wallet_first_trade_timestamp = AsyncMock(return_value=None)
        client.get_wallet_trade_count = AsyncMock(return_value=50)
        fs = run(det._score_account("0xabc"))
        assert fs.score == 0.0

    def test_api_exception_handled_gracefully(self):
        det, client = make_detector()
        client.get_wallet_first_trade_timestamp = AsyncMock(
            side_effect=Exception("network error"))
        fs = run(det._score_account("0xabc"))
        assert "lookup failed" in fs.detail


# ---------------------------------------------------------------------------
# score_trade (integration through all factors)
# ---------------------------------------------------------------------------

class TestScoreTrade:
    def test_below_min_size_returns_none(self):
        det, _ = make_detector()
        trade = {"usdcSize": "50", "conditionId": "c", "proxyWallet": "", "price": "0.5"}
        assert run(det.score_trade(trade, {})) is None

    def test_null_usdc_size_treated_as_zero(self):
        det, _ = make_detector()
        assert run(det.score_trade({"usdcSize": None}, {})) is None

    def test_missing_usdc_size_treated_as_zero(self):
        det, _ = make_detector()
        assert run(det.score_trade({}, {})) is None

    def test_score_below_threshold_returns_none(self):
        det, _ = make_detector()
        trade = {"usdcSize": "200", "conditionId": "c", "proxyWallet": "", "price": "0.5"}
        assert run(det.score_trade(trade, {})) is None

    def _cfg_for_score_trade(self, **overrides):
        """Pinned integration config so score_trade tests are self-contained."""
        cfg = make_test_cfg()
        for k, v in overrides.items():
            obj, attr = k.split("__", 1)
            setattr(getattr(cfg, obj), attr, v)
        return cfg

    def test_high_score_returns_anomaly_result(self):
        cfg = self._cfg_for_score_trade(
            scoring__alert_threshold=0.5,
            scoring__weight_by_liquidity=False,
            trade_size__large_trade_threshold_usd=100.0,
        )
        det, client = make_detector(cfg)
        client.hours_until_close.return_value = None
        trade = {"usdcSize": "200", "conditionId": "c", "proxyWallet": "", "price": "0.5"}
        result = run(det.score_trade(trade, {}))
        assert isinstance(result, AnomalyResult)

    def test_severity_low(self):
        # whale trade raw=3.0 < threshold*1.5=4.5 → LOW
        cfg = self._cfg_for_score_trade(
            scoring__alert_threshold=3.0,
            scoring__high_severity_threshold=10.0,
            scoring__weight_by_liquidity=False,
            scoring__max_single_factor_score=10.0,
        )
        det, client = make_detector(cfg)
        client.hours_until_close.return_value = None
        trade = {"usdcSize": "50000", "conditionId": "c", "proxyWallet": "", "price": "0.5"}
        result = run(det.score_trade(trade, {}))
        assert result is not None
        assert result.severity == "LOW"

    def test_severity_medium(self):
        # whale trade=3.0 >= threshold*1.5=3.0 → MEDIUM
        cfg = self._cfg_for_score_trade(
            scoring__alert_threshold=2.0,
            scoring__high_severity_threshold=10.0,
            scoring__weight_by_liquidity=False,
            scoring__max_single_factor_score=10.0,
        )
        det, client = make_detector(cfg)
        client.hours_until_close.return_value = None
        trade = {"usdcSize": "50000", "conditionId": "c", "proxyWallet": "", "price": "0.5"}
        result = run(det.score_trade(trade, {}))
        assert result is not None
        assert result.severity == "MEDIUM"

    def test_severity_high(self):
        # large(1.0) + whale(2.0) = 3.0 >= high_severity(3.0) → HIGH
        cfg = self._cfg_for_score_trade(
            scoring__alert_threshold=1.0,
            scoring__high_severity_threshold=3.0,
            scoring__weight_by_liquidity=False,
            scoring__max_single_factor_score=10.0,
        )
        det, client = make_detector(cfg)
        client.hours_until_close.return_value = None
        trade = {"usdcSize": "50000", "conditionId": "c", "proxyWallet": "", "price": "0.5"}
        result = run(det.score_trade(trade, {}))
        assert result is not None
        assert result.severity == "HIGH"

    def test_factor_clamped_to_max_single_factor_score(self):
        """No single factor can exceed max_single_factor_score."""
        cfg = self._cfg_for_score_trade(
            scoring__max_single_factor_score=0.5,
            scoring__alert_threshold=999.0,
            scoring__weight_by_liquidity=False,
        )
        det, client = make_detector(cfg)
        client.hours_until_close.return_value = None
        trade = {"usdcSize": "100000", "conditionId": "c", "proxyWallet": "", "price": "0.5"}
        assert run(det.score_trade(trade, {})) is None

    def test_liquidity_amplification_capped_at_2x(self):
        cfg = self._cfg_for_score_trade(
            scoring__alert_threshold=0.5,
            scoring__weight_by_liquidity=True,
            scoring__max_single_factor_score=1.0,
            volume__thin_market_volume_usd=1_000_000.0,
            volume__trade_volume_fraction_threshold=500.0,
            trade_size__large_trade_threshold_usd=100.0,
        )
        det, client = make_detector(cfg)
        client.hours_until_close.return_value = None
        trade = {"usdcSize": "200", "conditionId": "c", "proxyWallet": "", "price": "0.5"}
        result = run(det.score_trade(trade, {"volume24hr": "1"}))
        assert result is not None
        assert result.total_score == pytest.approx(2.0)

    def test_liquidity_weighting_skipped_when_disabled(self):
        cfg = self._cfg_for_score_trade(
            scoring__alert_threshold=0.5,
            scoring__weight_by_liquidity=False,
            scoring__max_single_factor_score=1.0,
            trade_size__large_trade_threshold_usd=100.0,
            volume__trade_volume_fraction_threshold=500.0,
        )
        det, client = make_detector(cfg)
        client.hours_until_close.return_value = None
        trade = {"usdcSize": "200", "conditionId": "c", "proxyWallet": "", "price": "0.5"}
        result = run(det.score_trade(trade, {"volume24hr": "1"}))
        assert result is not None
        assert result.total_score == pytest.approx(1.0)

    def test_liquidity_weighting_skipped_when_market_vol_zero(self):
        """weight_by_liquidity condition requires market_24h_vol > 0."""
        cfg = self._cfg_for_score_trade(
            scoring__alert_threshold=0.5,
            scoring__weight_by_liquidity=True,
            scoring__max_single_factor_score=1.0,
            trade_size__large_trade_threshold_usd=100.0,
        )
        det, client = make_detector(cfg)
        client.hours_until_close.return_value = None
        trade = {"usdcSize": "200", "conditionId": "c", "proxyWallet": "", "price": "0.5"}
        result = run(det.score_trade(trade, {"volume24hr": "0"}))
        assert result is not None
        assert result.total_score == pytest.approx(1.0)

    def test_result_total_score_is_rounded(self):
        cfg = self._cfg_for_score_trade(
            scoring__alert_threshold=0.5,
            scoring__weight_by_liquidity=False,
            trade_size__large_trade_threshold_usd=100.0,
        )
        det, client = make_detector(cfg)
        client.hours_until_close.return_value = None
        trade = {"usdcSize": "200", "conditionId": "c", "proxyWallet": "", "price": "0.5"}
        result = run(det.score_trade(trade, {}))
        assert result is not None
        assert result.total_score == round(result.total_score, 3)


# ---------------------------------------------------------------------------
# AnomalyResult.summary
# ---------------------------------------------------------------------------

class TestAnomalyResultSummary:
    def _make_result(self, severity="LOW", factors=None, trade=None, market=None):
        return AnomalyResult(
            trade=trade or {"usdcSize": "1000", "price": "0.7", "side": "BUY",
                            "proxyWallet": "0xabc"},
            market=market or {"question": "Will X happen?"},
            total_score=3.5,
            factors=factors or [FactorScore("trade_size", 1.5, "large trade")],
            severity=severity,
        )

    def test_includes_severity_tag(self):
        assert "[HIGH]" in self._make_result("HIGH").summary()
        assert "[LOW]" in self._make_result("LOW").summary()

    def test_includes_total_score(self):
        assert "3.50" in self._make_result().summary()

    def test_only_positive_factors_shown(self):
        factors = [
            FactorScore("trade_size", 1.5, "large"),
            FactorScore("timing", 0.0, "normal"),
        ]
        summary = self._make_result(factors=factors).summary()
        assert "trade_size" in summary
        assert "timing" not in summary

    def test_uses_question_field_for_market(self):
        assert "Will X happen?" in self._make_result().summary()

    def test_falls_back_to_condition_id(self):
        r = self._make_result(market={"conditionId": "0xcond"})
        assert "0xcond" in r.summary()

    def test_missing_trade_fields_show_question_mark(self):
        r = AnomalyResult(trade={}, market={}, total_score=1.0, factors=[], severity="LOW")
        assert "?" in r.summary()

    def test_wallet_address_shown(self):
        summary = self._make_result().summary()
        assert "0xabc" in summary
