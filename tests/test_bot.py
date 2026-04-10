"""
Unit tests for bot.py — load_config, AnomalyBot helpers.
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot import load_config, AnomalyBot
from config import AnomalyConfig


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_none_path_returns_default_config(self):
        cfg = load_config(None)
        assert isinstance(cfg, AnomalyConfig)
        assert cfg.scoring.alert_threshold == 3.0

    def _write_json(self, data):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, f)
        f.close()
        return f.name

    def test_overrides_sub_config_field(self):
        path = self._write_json({"scoring": {"alert_threshold": 1.5}})
        try:
            assert load_config(path).scoring.alert_threshold == 1.5
        finally:
            os.unlink(path)

    def test_other_sub_config_fields_unchanged(self):
        path = self._write_json({"scoring": {"alert_threshold": 1.5}})
        try:
            cfg = load_config(path)
            assert cfg.scoring.high_severity_threshold == 6.0  # default preserved
        finally:
            os.unlink(path)

    def test_multiple_sub_configs_overridden(self):
        data = {
            "trade_size": {"min_trade_size_usd": 500.0},
            "timing": {"late_trade_window_hours": 12.0},
        }
        path = self._write_json(data)
        try:
            cfg = load_config(path)
            assert cfg.trade_size.min_trade_size_usd == 500.0
            assert cfg.timing.late_trade_window_hours == 12.0
        finally:
            os.unlink(path)

    def test_unknown_top_level_key_ignored(self):
        path = self._write_json({"no_such_section": {"foo": 1}})
        try:
            cfg = load_config(path)  # must not raise
            assert isinstance(cfg, AnomalyConfig)
        finally:
            os.unlink(path)

    def test_unknown_field_within_sub_config_ignored(self):
        path = self._write_json({"scoring": {"nonexistent_field": 99}})
        try:
            cfg = load_config(path)
            assert cfg.scoring.alert_threshold == 3.0  # unchanged
        finally:
            os.unlink(path)

    def test_all_sub_sections_settable(self):
        data = {
            "account": {"new_account_age_days": 7},
            "volume": {"volume_spike_multiplier": 5.0},
            "trade_size": {"whale_trade_threshold_usd": 100_000.0},
            "price_impact": {"min_price_impact": 0.02},
            "timing": {"early_trade_window_hours": 2.0},
            "scoring": {"max_single_factor_score": 5.0},
        }
        path = self._write_json(data)
        try:
            cfg = load_config(path)
            assert cfg.account.new_account_age_days == 7
            assert cfg.volume.volume_spike_multiplier == 5.0
            assert cfg.trade_size.whale_trade_threshold_usd == 100_000.0
            assert cfg.price_impact.min_price_impact == 0.02
            assert cfg.timing.early_trade_window_hours == 2.0
            assert cfg.scoring.max_single_factor_score == 5.0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# AnomalyBot._refresh_markets
# ---------------------------------------------------------------------------

class TestRefreshMarkets:
    def _bot(self, ignored=None):
        cfg = AnomalyConfig()
        cfg.ignored_markets = ignored or []
        return AnomalyBot(cfg)

    def test_caches_markets_by_condition_id(self):
        bot = self._bot()
        client = MagicMock()
        client.get_all_active_markets = AsyncMock(return_value=[
            {"conditionId": "cid1", "slug": "s1"},
        ])
        run(bot._refresh_markets(client))
        assert "cid1" in bot._market_cache

    def test_filters_ignored_condition_id(self):
        bot = self._bot(ignored=["bad_cid"])
        client = MagicMock()
        client.get_all_active_markets = AsyncMock(return_value=[
            {"conditionId": "bad_cid", "slug": "ok"},
            {"conditionId": "good_cid", "slug": "ok2"},
        ])
        run(bot._refresh_markets(client))
        assert "bad_cid" not in bot._market_cache
        assert "good_cid" in bot._market_cache

    def test_filters_ignored_slug(self):
        bot = self._bot(ignored=["bad-slug"])
        client = MagicMock()
        client.get_all_active_markets = AsyncMock(return_value=[
            {"conditionId": "cid1", "slug": "bad-slug"},
            {"conditionId": "cid2", "slug": "good-slug"},
        ])
        run(bot._refresh_markets(client))
        assert "cid1" not in bot._market_cache
        assert "cid2" in bot._market_cache

    def test_skips_markets_without_condition_id(self):
        bot = self._bot()
        client = MagicMock()
        client.get_all_active_markets = AsyncMock(return_value=[
            {"slug": "no-cid"},
            {"conditionId": "cid_ok", "slug": "ok"},
        ])
        run(bot._refresh_markets(client))
        assert len(bot._market_cache) == 1
        assert "cid_ok" in bot._market_cache

    def test_skips_markets_with_empty_condition_id(self):
        bot = self._bot()
        client = MagicMock()
        client.get_all_active_markets = AsyncMock(return_value=[
            {"conditionId": "", "slug": "empty-cid"},
        ])
        run(bot._refresh_markets(client))
        assert bot._market_cache == {}

    def test_api_failure_does_not_crash(self):
        bot = self._bot()
        client = MagicMock()
        client.get_all_active_markets = AsyncMock(side_effect=Exception("network error"))
        run(bot._refresh_markets(client))  # must not raise
        assert bot._market_cache == {}

    def test_replaces_existing_cache_on_refresh(self):
        bot = self._bot()
        bot._market_cache = {"stale_cid": {"conditionId": "stale_cid"}}
        client = MagicMock()
        client.get_all_active_markets = AsyncMock(return_value=[
            {"conditionId": "new_cid", "slug": "new"},
        ])
        run(bot._refresh_markets(client))
        assert "stale_cid" not in bot._market_cache
        assert "new_cid" in bot._market_cache


# ---------------------------------------------------------------------------
# AnomalyBot._scan_market
# ---------------------------------------------------------------------------

class TestScanMarket:
    def _setup(self):
        bot = AnomalyBot(AnomalyConfig())
        client = MagicMock()
        detector = MagicMock()
        detector.score_trade = AsyncMock(return_value=None)
        return bot, client, detector

    def _market(self, yes_token_id=None):
        tokens = []
        if yes_token_id:
            tokens = [{"token_id": yes_token_id, "outcome": "Yes"}]
        return {"conditionId": "cid", "tokens": tokens}

    def test_deduplication_by_transaction_hash(self):
        bot, client, detector = self._setup()
        trade = {"transactionHash": "tx1", "price": "0.5", "usdcSize": "200"}
        client.get_trades = AsyncMock(return_value=[trade, trade])
        run(bot._scan_market(client, detector, self._market()))
        assert detector.score_trade.call_count == 1

    def test_deduplication_by_id_fallback(self):
        bot, client, detector = self._setup()
        trade = {"id": "id_abc", "price": "0.5", "usdcSize": "200"}
        client.get_trades = AsyncMock(return_value=[trade, trade])
        run(bot._scan_market(client, detector, self._market()))
        assert detector.score_trade.call_count == 1

    def test_trades_without_hash_or_id_scored_every_time(self):
        """Trades with no identifier cannot be deduplicated."""
        bot, client, detector = self._setup()
        trade = {"price": "0.5", "usdcSize": "200"}  # no tx hash, no id
        client.get_trades = AsyncMock(return_value=[trade, trade])
        run(bot._scan_market(client, detector, self._market()))
        assert detector.score_trade.call_count == 2

    def test_seen_tx_populated_after_scan(self):
        bot, client, detector = self._setup()
        trade = {"transactionHash": "tx_abc", "price": "0.5", "usdcSize": "200"}
        client.get_trades = AsyncMock(return_value=[trade])
        run(bot._scan_market(client, detector, self._market()))
        assert "tx_abc" in bot._seen_tx

    def test_already_seen_tx_not_rescored(self):
        bot, client, detector = self._setup()
        bot._seen_tx.add("tx_old")
        trade = {"transactionHash": "tx_old", "price": "0.5", "usdcSize": "200"}
        client.get_trades = AsyncMock(return_value=[trade])
        run(bot._scan_market(client, detector, self._market()))
        detector.score_trade.assert_not_called()

    def test_pre_price_read_from_cache(self):
        bot, client, detector = self._setup()
        bot._last_price["yes_tok"] = 0.55
        trade = {"transactionHash": "tx1", "price": "0.65", "usdcSize": "200"}
        client.get_trades = AsyncMock(return_value=[trade])
        run(bot._scan_market(client, detector, self._market("yes_tok")))
        _args, _kwargs = detector.score_trade.call_args
        assert _args[2] == 0.55  # pre_trade_price

    def test_pre_price_none_when_not_cached(self):
        bot, client, detector = self._setup()
        trade = {"transactionHash": "tx1", "price": "0.65", "usdcSize": "200"}
        client.get_trades = AsyncMock(return_value=[trade])
        run(bot._scan_market(client, detector, self._market("yes_tok")))
        _args, _kwargs = detector.score_trade.call_args
        assert _args[2] is None

    def test_post_price_updates_cache(self):
        bot, client, detector = self._setup()
        trade = {"transactionHash": "tx1", "price": "0.70", "usdcSize": "200"}
        client.get_trades = AsyncMock(return_value=[trade])
        run(bot._scan_market(client, detector, self._market("yes_tok")))
        assert bot._last_price["yes_tok"] == pytest.approx(0.70)

    def test_zero_post_price_does_not_update_cache(self):
        bot, client, detector = self._setup()
        bot._last_price["yes_tok"] = 0.60
        trade = {"transactionHash": "tx1", "price": "0", "usdcSize": "200"}
        client.get_trades = AsyncMock(return_value=[trade])
        run(bot._scan_market(client, detector, self._market("yes_tok")))
        assert bot._last_price["yes_tok"] == 0.60  # unchanged

    def test_alert_fired_on_anomaly_result(self):
        bot, client, detector = self._setup()
        mock_result = MagicMock()
        mock_result.summary = MagicMock(return_value="summary")
        detector.score_trade = AsyncMock(return_value=mock_result)
        trade = {"transactionHash": "tx1", "price": "0.7", "usdcSize": "50000"}
        client.get_trades = AsyncMock(return_value=[trade])
        with patch("bot.alert") as mock_alert:
            run(bot._scan_market(client, detector, self._market()))
            mock_alert.assert_called_once_with(mock_result)

    def test_no_alert_when_score_trade_returns_none(self):
        bot, client, detector = self._setup()
        trade = {"transactionHash": "tx1", "price": "0.5", "usdcSize": "200"}
        client.get_trades = AsyncMock(return_value=[trade])
        with patch("bot.alert") as mock_alert:
            run(bot._scan_market(client, detector, self._market()))
            mock_alert.assert_not_called()

    def test_fetch_error_returns_early_without_crash(self):
        bot, client, detector = self._setup()
        client.get_trades = AsyncMock(side_effect=Exception("timeout"))
        run(bot._scan_market(client, detector, self._market()))
        detector.score_trade.assert_not_called()

    def test_no_yes_token_pre_price_is_none(self):
        """Market with no 'Yes' outcome token → pre_price stays None."""
        bot, client, detector = self._setup()
        trade = {"transactionHash": "tx1", "price": "0.5", "usdcSize": "200"}
        market = {
            "conditionId": "cid",
            "tokens": [{"token_id": "no_tok", "outcome": "No"}],
        }
        client.get_trades = AsyncMock(return_value=[trade])
        run(bot._scan_market(client, detector, market))
        _args, _ = detector.score_trade.call_args
        assert _args[2] is None


# ---------------------------------------------------------------------------
# AnomalyBot._poll — seen_tx size management
# ---------------------------------------------------------------------------

class TestPollSeenTxBounded:
    def _bot_with_cache(self, seen_count):
        bot = AnomalyBot(AnomalyConfig())
        bot._market_cache = {"cid": {"conditionId": "cid", "slug": "s"}}
        bot._seen_tx = set(str(i) for i in range(seen_count))
        return bot

    def test_seen_tx_cleared_when_over_limit(self):
        bot = self._bot_with_cache(100_001)
        client = MagicMock()
        detector = MagicMock()
        with patch.object(bot, "_refresh_markets", new=AsyncMock()):
            with patch.object(bot, "_scan_market", new=AsyncMock()):
                run(bot._poll(client, detector))
        assert len(bot._seen_tx) == 0

    def test_seen_tx_not_cleared_below_limit(self):
        bot = self._bot_with_cache(50)
        client = MagicMock()
        detector = MagicMock()
        with patch.object(bot, "_refresh_markets", new=AsyncMock()):
            with patch.object(bot, "_scan_market", new=AsyncMock()):
                run(bot._poll(client, detector))
        assert len(bot._seen_tx) == 50  # untouched

    def test_seen_tx_exactly_at_limit_not_cleared(self):
        """The condition is > 100_000, so exactly 100_000 is NOT cleared."""
        bot = self._bot_with_cache(100_000)
        client = MagicMock()
        detector = MagicMock()
        with patch.object(bot, "_refresh_markets", new=AsyncMock()):
            with patch.object(bot, "_scan_market", new=AsyncMock()):
                run(bot._poll(client, detector))
        assert len(bot._seen_tx) == 100_000


# ---------------------------------------------------------------------------
# AnomalyBot._poll — watched_markets filter
# ---------------------------------------------------------------------------

class TestPollWatchedMarkets:
    def test_only_watched_markets_scanned(self):
        cfg = AnomalyConfig()
        cfg.watched_markets = ["cid_a"]
        bot = AnomalyBot(cfg)
        bot._market_cache = {
            "cid_a": {"conditionId": "cid_a", "slug": "sa"},
            "cid_b": {"conditionId": "cid_b", "slug": "sb"},
        }

        scanned = []

        async def fake_scan(client, detector, market):
            scanned.append(market["conditionId"])

        client = MagicMock()
        detector = MagicMock()
        with patch.object(bot, "_refresh_markets", new=AsyncMock()):
            with patch.object(bot, "_scan_market", new=fake_scan):
                run(bot._poll(client, detector))

        assert scanned == ["cid_a"]

    def test_watched_by_slug_included(self):
        cfg = AnomalyConfig()
        cfg.watched_markets = ["slug_b"]
        bot = AnomalyBot(cfg)
        bot._market_cache = {
            "cid_a": {"conditionId": "cid_a", "slug": "slug_a"},
            "cid_b": {"conditionId": "cid_b", "slug": "slug_b"},
        }

        scanned = []

        async def fake_scan(client, detector, market):
            scanned.append(market["conditionId"])

        client = MagicMock()
        detector = MagicMock()
        with patch.object(bot, "_refresh_markets", new=AsyncMock()):
            with patch.object(bot, "_scan_market", new=fake_scan):
                run(bot._poll(client, detector))

        assert "cid_b" in scanned
        assert "cid_a" not in scanned

    def test_empty_watched_markets_scans_all(self):
        cfg = AnomalyConfig()
        cfg.watched_markets = []  # empty → all
        bot = AnomalyBot(cfg)
        bot._market_cache = {
            "cid_a": {"conditionId": "cid_a", "slug": "sa"},
            "cid_b": {"conditionId": "cid_b", "slug": "sb"},
        }

        scanned = []

        async def fake_scan(client, detector, market):
            scanned.append(market["conditionId"])

        client = MagicMock()
        detector = MagicMock()
        with patch.object(bot, "_refresh_markets", new=AsyncMock()):
            with patch.object(bot, "_scan_market", new=fake_scan):
                run(bot._poll(client, detector))

        assert set(scanned) == {"cid_a", "cid_b"}


# ---------------------------------------------------------------------------
# AnomalyBot._collect_tokens
# ---------------------------------------------------------------------------

class TestCollectTokens:
    def _bot_with_cache(self, market_cache, watched=None):
        cfg = AnomalyConfig()
        if watched is not None:
            cfg.watched_markets = watched
        bot = AnomalyBot(cfg)
        bot._market_cache = market_cache
        return bot

    def _market(self, cid, slug, tokens):
        return {"conditionId": cid, "slug": slug, "tokens": tokens}

    def _tok(self, tid, outcome="Yes"):
        return {"token_id": tid, "outcome": outcome}

    def test_returns_all_token_ids(self):
        cache = {
            "cid1": self._market("cid1", "s1", [self._tok("t1"), self._tok("t2", "No")]),
            "cid2": self._market("cid2", "s2", [self._tok("t3")]),
        }
        bot = self._bot_with_cache(cache)
        ids, mapping = bot._collect_tokens()
        assert set(ids) == {"t1", "t2", "t3"}

    def test_each_token_maps_to_correct_market(self):
        cache = {
            "cid1": self._market("cid1", "s1", [self._tok("t1")]),
            "cid2": self._market("cid2", "s2", [self._tok("t2")]),
        }
        bot = self._bot_with_cache(cache)
        _, mapping = bot._collect_tokens()
        assert mapping["t1"]["conditionId"] == "cid1"
        assert mapping["t2"]["conditionId"] == "cid2"

    def test_watched_markets_filters_by_condition_id(self):
        cache = {
            "cid_a": self._market("cid_a", "sa", [self._tok("ta")]),
            "cid_b": self._market("cid_b", "sb", [self._tok("tb")]),
        }
        bot = self._bot_with_cache(cache, watched=["cid_a"])
        ids, _ = bot._collect_tokens()
        assert "ta" in ids
        assert "tb" not in ids

    def test_watched_markets_filters_by_slug(self):
        cache = {
            "cid_a": self._market("cid_a", "slug_a", [self._tok("ta")]),
            "cid_b": self._market("cid_b", "slug_b", [self._tok("tb")]),
        }
        bot = self._bot_with_cache(cache, watched=["slug_b"])
        ids, _ = bot._collect_tokens()
        assert "tb" in ids
        assert "ta" not in ids

    def test_empty_cache_returns_empty(self):
        bot = self._bot_with_cache({})
        ids, mapping = bot._collect_tokens()
        assert ids == []
        assert mapping == {}

    def test_market_without_tokens_key(self):
        cache = {"cid1": {"conditionId": "cid1", "slug": "s1"}}  # no "tokens"
        bot = self._bot_with_cache(cache)
        ids, _ = bot._collect_tokens()
        assert ids == []

    def test_token_without_token_id_skipped(self):
        cache = {"cid1": self._market("cid1", "s1", [{"outcome": "Yes"}])}
        bot = self._bot_with_cache(cache)
        ids, _ = bot._collect_tokens()
        assert ids == []


# ---------------------------------------------------------------------------
# AnomalyBot._process_streamed_trade
# ---------------------------------------------------------------------------

class TestProcessStreamedTrade:
    def _setup(self):
        bot = AnomalyBot(AnomalyConfig())
        detector = MagicMock()
        detector.score_trade = AsyncMock(return_value=None)
        return bot, detector

    def _trade(self, token_id="tok1", price="0.6", usdc="60", cid="cid1", ws_id=None):
        return {
            "usdcSize": usdc,
            "price": price,
            "conditionId": cid,
            "proxyWallet": "",
            "_token_id": token_id,
            "_ws_id": ws_id or f"{token_id}_1000_{price}_{usdc}",
        }

    def test_deduplication_by_ws_id(self):
        bot, detector = self._setup()
        trade = self._trade()
        market = {"conditionId": "cid1"}
        token_to_market = {"tok1": market}

        run(bot._process_streamed_trade(trade, token_to_market, detector))
        run(bot._process_streamed_trade(trade, token_to_market, detector))
        assert detector.score_trade.call_count == 1

    def test_token_to_market_lookup(self):
        bot, detector = self._setup()
        market = {"conditionId": "cid1"}
        token_to_market = {"tok1": market}
        trade = self._trade(token_id="tok1")

        run(bot._process_streamed_trade(trade, token_to_market, detector))
        _args, _ = detector.score_trade.call_args
        assert _args[1] is market

    def test_falls_back_to_market_cache_by_condition_id(self):
        bot, detector = self._setup()
        market = {"conditionId": "cid1"}
        bot._market_cache = {"cid1": market}
        trade = self._trade(token_id="unknown_tok", cid="cid1")

        run(bot._process_streamed_trade(trade, {}, detector))
        _args, _ = detector.score_trade.call_args
        assert _args[1] is market

    def test_uses_empty_dict_when_market_not_found(self):
        bot, detector = self._setup()
        trade = self._trade(token_id="tok_x", cid="unknown_cid")

        run(bot._process_streamed_trade(trade, {}, detector))
        _args, _ = detector.score_trade.call_args
        assert _args[1] == {}

    def test_pre_price_from_last_price_cache(self):
        bot, detector = self._setup()
        bot._last_price["tok1"] = 0.55
        trade = self._trade(token_id="tok1", price="0.65")

        run(bot._process_streamed_trade(trade, {}, detector))
        _args, _ = detector.score_trade.call_args
        assert _args[2] == pytest.approx(0.55)

    def test_post_price_updates_last_price_cache(self):
        bot, detector = self._setup()
        trade = self._trade(token_id="tok1", price="0.70")

        run(bot._process_streamed_trade(trade, {}, detector))
        assert bot._last_price["tok1"] == pytest.approx(0.70)

    def test_zero_post_price_does_not_update_cache(self):
        bot, detector = self._setup()
        bot._last_price["tok1"] = 0.60
        trade = self._trade(token_id="tok1", price="0")

        run(bot._process_streamed_trade(trade, {}, detector))
        assert bot._last_price["tok1"] == pytest.approx(0.60)

    def test_alert_called_on_anomaly_result(self):
        bot, detector = self._setup()
        mock_result = MagicMock()
        detector.score_trade = AsyncMock(return_value=mock_result)
        trade = self._trade()

        with patch("bot.alert") as mock_alert:
            run(bot._process_streamed_trade(trade, {}, detector))
            mock_alert.assert_called_once_with(mock_result)

    def test_seen_tx_cleared_when_over_limit(self):
        bot, detector = self._setup()
        bot._seen_tx = set(str(i) for i in range(100_001))
        trade = self._trade(ws_id="brand_new_id")

        run(bot._process_streamed_trade(trade, {}, detector))
        assert len(bot._seen_tx) == 0  # cleared before adding new id
