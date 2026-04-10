"""
Unit tests for polymarket_client.py — PolymarketClient.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from polymarket_client import PolymarketClient


def run(coro):
    return asyncio.run(coro)


def make_client(response_data):
    """Return a PolymarketClient whose HTTP session always returns response_data."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value=response_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    return PolymarketClient(session=mock_session)


def make_paged_client(pages):
    """Return a PolymarketClient whose json() returns successive pages."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(side_effect=pages)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    return PolymarketClient(session=mock_session)


# ---------------------------------------------------------------------------
# market_end_timestamp (static)
# ---------------------------------------------------------------------------

class TestMarketEndTimestamp:
    def test_valid_end_date(self):
        ts = PolymarketClient.market_end_timestamp({"endDate": "2026-12-31T00:00:00Z"})
        assert ts is not None
        assert isinstance(ts, int)

    def test_end_date_iso_fallback_key(self):
        ts = PolymarketClient.market_end_timestamp({"end_date_iso": "2026-06-01T00:00:00Z"})
        assert ts is not None

    def test_end_date_preferred_over_iso(self):
        from datetime import datetime, timezone
        market = {"endDate": "2026-12-31T00:00:00Z", "end_date_iso": "2000-01-01T00:00:00Z"}
        ts = PolymarketClient.market_end_timestamp(market)
        expected = int(datetime(2026, 12, 31, tzinfo=timezone.utc).timestamp())
        assert ts == expected

    def test_missing_end_date_returns_none(self):
        assert PolymarketClient.market_end_timestamp({}) is None

    def test_none_end_date_returns_none(self):
        assert PolymarketClient.market_end_timestamp({"endDate": None}) is None

    def test_invalid_format_returns_none(self):
        assert PolymarketClient.market_end_timestamp({"endDate": "not-a-date"}) is None

    def test_empty_string_end_date_returns_none(self):
        assert PolymarketClient.market_end_timestamp({"endDate": ""}) is None


# ---------------------------------------------------------------------------
# hours_until_close (static)
# ---------------------------------------------------------------------------

class TestHoursUntilClose:
    def test_future_market_positive_hours(self):
        h = PolymarketClient.hours_until_close({"endDate": "2099-01-01T00:00:00Z"})
        assert h is not None
        assert h > 0

    def test_past_market_negative_hours(self):
        h = PolymarketClient.hours_until_close({"endDate": "2000-01-01T00:00:00Z"})
        assert h is not None
        assert h < 0

    def test_missing_end_date_returns_none(self):
        assert PolymarketClient.hours_until_close({}) is None


# ---------------------------------------------------------------------------
# get_active_markets
# ---------------------------------------------------------------------------

class TestGetActiveMarkets:
    def test_list_response_returned_as_is(self):
        data = [{"conditionId": "a"}, {"conditionId": "b"}]
        assert run(make_client(data).get_active_markets()) == data

    def test_dict_with_markets_key_unwrapped(self):
        data = {"markets": [{"conditionId": "a"}]}
        assert run(make_client(data).get_active_markets()) == [{"conditionId": "a"}]

    def test_empty_list_response(self):
        assert run(make_client([]).get_active_markets()) == []

    def test_dict_without_markets_key_returns_empty(self):
        assert run(make_client({"other": "data"}).get_active_markets()) == []


# ---------------------------------------------------------------------------
# get_all_active_markets  (pagination)
# ---------------------------------------------------------------------------

class TestGetAllActiveMarkets:
    def test_stops_on_empty_page(self):
        pages = [
            [{"conditionId": "a"}, {"conditionId": "b"}],
            [],
        ]
        result = run(make_paged_client(pages).get_all_active_markets(page_size=100))
        assert len(result) == 2

    def test_stops_when_page_smaller_than_page_size(self):
        pages = [
            [{"conditionId": str(i)} for i in range(50)],  # 50 < 100
        ]
        result = run(make_paged_client(pages).get_all_active_markets(page_size=100))
        assert len(result) == 50

    def test_aggregates_multiple_full_pages(self):
        pages = [
            [{"conditionId": str(i)} for i in range(10)],
            [{"conditionId": str(i + 10)} for i in range(10)],
            [],
        ]
        result = run(make_paged_client(pages).get_all_active_markets(page_size=10))
        assert len(result) == 20


# ---------------------------------------------------------------------------
# get_trades
# ---------------------------------------------------------------------------

class TestGetTrades:
    def test_list_response(self):
        data = [{"transactionHash": "tx1"}, {"transactionHash": "tx2"}]
        assert run(make_client(data).get_trades()) == data

    def test_dict_with_trades_key_unwrapped(self):
        data = {"trades": [{"transactionHash": "tx1"}]}
        assert run(make_client(data).get_trades()) == [{"transactionHash": "tx1"}]

    def test_empty_list(self):
        assert run(make_client([]).get_trades()) == []


# ---------------------------------------------------------------------------
# get_wallet_activity
# ---------------------------------------------------------------------------

class TestGetWalletActivity:
    def test_list_response_returned_as_is(self):
        data = [{"type": "TRADE"}, {"type": "ORDER"}]
        assert run(make_client(data).get_wallet_activity("0xabc")) == data

    def test_non_list_response_returns_empty_list(self):
        """If the API returns a non-list (e.g. error dict), return []."""
        assert run(make_client({"error": "not found"}).get_wallet_activity("0x")) == []

    def test_empty_list(self):
        assert run(make_client([]).get_wallet_activity("0xabc")) == []


# ---------------------------------------------------------------------------
# get_wallet_trade_count
# ---------------------------------------------------------------------------

class TestGetWalletTradeCount:
    def test_counts_only_trade_type_activities(self):
        data = [
            {"type": "TRADE"},
            {"type": "ORDER"},
            {"type": "TRADE"},
            {"type": "TRANSFER"},
            {"type": "trade"},   # lowercase — NOT matched
        ]
        assert run(make_client(data).get_wallet_trade_count("0xabc")) == 2

    def test_empty_activity_returns_zero(self):
        assert run(make_client([]).get_wallet_trade_count("0xabc")) == 0

    def test_no_trade_type_entries_returns_zero(self):
        data = [{"type": "ORDER"}, {"type": "TRANSFER"}]
        assert run(make_client(data).get_wallet_trade_count("0xabc")) == 0


# ---------------------------------------------------------------------------
# get_wallet_first_trade_timestamp
# ---------------------------------------------------------------------------

class TestGetWalletFirstTradeTimestamp:
    def test_returns_timestamp_of_first_item(self):
        data = [{"timestamp": 1700000000}, {"timestamp": 1700010000}]
        ts = run(make_client(data).get_wallet_first_trade_timestamp("0xabc"))
        assert ts == 1700000000

    def test_empty_list_returns_none(self):
        ts = run(make_client([]).get_wallet_first_trade_timestamp("0xabc"))
        assert ts is None

    def test_non_list_response_returns_none(self):
        ts = run(make_client({"data": []}).get_wallet_first_trade_timestamp("0xabc"))
        assert ts is None

    def test_item_without_timestamp_returns_none(self):
        ts = run(make_client([{"id": "x"}]).get_wallet_first_trade_timestamp("0xabc"))
        assert ts is None


# ---------------------------------------------------------------------------
# get_midpoint
# ---------------------------------------------------------------------------

class TestGetMidpoint:
    def test_valid_mid_value(self):
        result = run(make_client({"mid": "0.65"}).get_midpoint("tok"))
        assert result == pytest.approx(0.65)

    def test_numeric_mid_value(self):
        result = run(make_client({"mid": 0.5}).get_midpoint("tok"))
        assert result == pytest.approx(0.5)

    def test_none_mid_returns_none(self):
        assert run(make_client({"mid": None}).get_midpoint("tok")) is None

    def test_missing_mid_key_returns_none(self):
        assert run(make_client({}).get_midpoint("tok")) is None


# ---------------------------------------------------------------------------
# get_last_trade_price
# ---------------------------------------------------------------------------

class TestGetLastTradePrice:
    def test_valid_price(self):
        result = run(make_client({"price": "0.72"}).get_last_trade_price("tok"))
        assert result == pytest.approx(0.72)

    def test_none_price_returns_none(self):
        assert run(make_client({"price": None}).get_last_trade_price("tok")) is None

    def test_missing_price_key_returns_none(self):
        assert run(make_client({}).get_last_trade_price("tok")) is None


# ---------------------------------------------------------------------------
# Context manager / session ownership
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _normalise_ws_trade (static)
# ---------------------------------------------------------------------------

class TestNormaliseWsTrade:
    def test_valid_event_returns_dict(self):
        event = {
            "event_type": "last_trade_price",
            "asset": "tok123",
            "price": 0.75,
            "size": 100,
            "side": "BUY",
            "timestamp": 1700000000,
            "conditionId": "cid1",
        }
        trade = PolymarketClient._normalise_ws_trade(event)
        assert trade is not None
        assert trade["price"] == "0.75"
        assert trade["side"] == "BUY"
        assert trade["conditionId"] == "cid1"
        assert trade["proxyWallet"] == ""
        assert trade["_token_id"] == "tok123"

    def test_usdcsize_is_price_times_size(self):
        trade = PolymarketClient._normalise_ws_trade(
            {"asset": "t", "price": 0.5, "size": 200, "timestamp": 1}
        )
        assert trade is not None
        assert float(trade["usdcSize"]) == pytest.approx(100.0)

    def test_ws_id_is_unique_per_event(self):
        base = {"asset": "t", "price": 0.5, "size": 100, "timestamp": 1000}
        t1 = PolymarketClient._normalise_ws_trade({**base})
        t2 = PolymarketClient._normalise_ws_trade({**base, "timestamp": 1001})
        assert t1["_ws_id"] != t2["_ws_id"]

    def test_zero_price_returns_none(self):
        assert PolymarketClient._normalise_ws_trade(
            {"asset": "t", "price": 0, "size": 100, "timestamp": 1}
        ) is None

    def test_zero_size_returns_none(self):
        assert PolymarketClient._normalise_ws_trade(
            {"asset": "t", "price": 0.5, "size": 0, "timestamp": 1}
        ) is None

    def test_missing_price_returns_none(self):
        assert PolymarketClient._normalise_ws_trade(
            {"asset": "t", "size": 100, "timestamp": 1}
        ) is None

    def test_non_numeric_price_returns_none(self):
        assert PolymarketClient._normalise_ws_trade(
            {"asset": "t", "price": "bad", "size": 100, "timestamp": 1}
        ) is None

    def test_string_numeric_fields_accepted(self):
        """WS may send numbers as strings."""
        trade = PolymarketClient._normalise_ws_trade(
            {"asset": "t", "price": "0.6", "size": "50", "timestamp": "1000"}
        )
        assert trade is not None
        assert float(trade["usdcSize"]) == pytest.approx(30.0)

    def test_missing_optional_fields_use_empty_string(self):
        trade = PolymarketClient._normalise_ws_trade(
            {"price": 0.5, "size": 100, "timestamp": 1}
        )
        assert trade is not None
        assert trade["side"] == ""
        assert trade["conditionId"] == ""
        assert trade["_token_id"] == ""


# ---------------------------------------------------------------------------
# stream_trades  (mocked WebSocket)
# ---------------------------------------------------------------------------

class TestStreamTrades:
    """
    stream_trades runs a `while True` reconnection loop, so tests use
    asyncio.wait_for to bound execution.  All mock operations are instant
    so trades are collected before the timeout fires; the timeout just
    interrupts the reconnect sleep.
    """

    def _make_ws_client(self, payloads):
        """
        Build a PolymarketClient whose WS yields one TEXT message per payload
        then exhausts.  `payloads` is a list of dicts or lists (raw JSON bodies).
        """
        import aiohttp as _aio
        import json as _json

        class FakeWS:
            def __init__(self):
                self._iter = iter(payloads)

            async def send_json(self, _):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    payload = next(self._iter)
                    msg = MagicMock()
                    msg.type = _aio.WSMsgType.TEXT
                    msg.data = _json.dumps(payload)
                    return msg
                except StopIteration:
                    raise StopAsyncIteration

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

        mock_session = MagicMock()
        # Return a fresh FakeWS on each call so reconnects don't re-deliver messages
        mock_session.ws_connect = MagicMock(side_effect=lambda *a, **k: FakeWS())
        return PolymarketClient(session=mock_session)

    @staticmethod
    def _collect(client, token_ids, timeout=0.5):
        """Run stream_trades, collecting trades until timeout interrupts the reconnect sleep."""
        async def _inner():
            trades = []
            async def _gather():
                async for t in client.stream_trades(token_ids):
                    trades.append(t)
            try:
                await asyncio.wait_for(_gather(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
            return trades
        return run(_inner())

    def test_yields_trade_for_last_trade_price_event(self):
        event = [{
            "event_type": "last_trade_price",
            "asset": "tok1", "price": 0.7, "size": 100,
            "side": "BUY", "timestamp": 1000, "conditionId": "cid1",
        }]
        trades = self._collect(self._make_ws_client([event]), ["tok1"])
        assert len(trades) == 1
        assert trades[0]["conditionId"] == "cid1"
        assert float(trades[0]["usdcSize"]) == pytest.approx(70.0)

    def test_skips_non_trade_events(self):
        payloads = [
            [{"event_type": "book", "asset": "tok1"}],
            [{"event_type": "last_trade_price", "asset": "tok1",
              "price": 0.5, "size": 50, "timestamp": 1, "conditionId": "c1"}],
        ]
        trades = self._collect(self._make_ws_client(payloads), ["tok1"])
        assert len(trades) == 1

    def test_handles_single_object_payload(self):
        """WS may send a single event dict instead of a list."""
        single = {
            "event_type": "last_trade_price",
            "asset": "tok1", "price": 0.6, "size": 80,
            "timestamp": 2, "conditionId": "cid2",
        }
        trades = self._collect(self._make_ws_client([single]), ["tok1"])
        assert len(trades) == 1

    def test_empty_token_list_yields_nothing(self):
        client = self._make_ws_client([])

        async def collect():
            return [t async for t in client.stream_trades([])]

        assert run(collect()) == []

    def test_skips_events_with_invalid_price(self):
        payloads = [[
            {"event_type": "last_trade_price", "asset": "t",
             "price": 0, "size": 100, "timestamp": 1},
        ]]
        trades = self._collect(self._make_ws_client(payloads), ["t"])
        assert trades == []

    def test_multiple_trades_in_one_message(self):
        payloads = [[
            {"event_type": "last_trade_price", "asset": "t", "price": 0.5,
             "size": 100, "timestamp": 1, "conditionId": "c1"},
            {"event_type": "last_trade_price", "asset": "t", "price": 0.6,
             "size": 50, "timestamp": 2, "conditionId": "c1"},
        ]]
        trades = self._collect(self._make_ws_client(payloads), ["t"])
        assert len(trades) == 2


# ---------------------------------------------------------------------------
# Context manager / session ownership
# ---------------------------------------------------------------------------

class TestClientSessionOwnership:
    def test_external_session_not_closed_on_exit(self):
        """When the caller owns the session, client.__aexit__ must not close it."""
        mock_session = MagicMock()
        mock_session.close = AsyncMock()
        client = PolymarketClient(session=mock_session)

        async def _run():
            async with client:
                pass

        run(_run())
        mock_session.close.assert_not_called()
