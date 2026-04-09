"""
PolymarketClient — thin async wrapper around Polymarket's public APIs.

APIs used (all public, no auth required):
  Gamma API  : market/event discovery
  Data API   : trades, activity, positions
  CLOB API   : orderbook, price history, midpoints
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

import aiohttp


GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


class PolymarketClient:
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self):
        if self._owns_session:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_):
        if self._owns_session and self._session:
            await self._session.close()

    async def _get(self, base: str, path: str, params: dict = None) -> dict | list:
        url = f"{base}{path}"
        async with self._session.get(url, params=params or {}) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ------------------------------------------------------------------ #
    # Gamma API — market discovery
    # ------------------------------------------------------------------ #

    async def get_active_markets(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Return a page of active markets."""
        data = await self._get(GAMMA_BASE, "/markets", {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
        })
        return data if isinstance(data, list) else data.get("markets", [])

    async def get_market(self, condition_id: str) -> dict:
        """Return a single market by condition_id."""
        return await self._get(GAMMA_BASE, f"/markets/{condition_id}")

    async def get_all_active_markets(self, page_size: int = 100) -> list[dict]:
        """Paginate through all active markets."""
        markets, offset = [], 0
        while True:
            page = await self.get_active_markets(limit=page_size, offset=offset)
            if not page:
                break
            markets.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return markets

    # ------------------------------------------------------------------ #
    # Data API — trades & activity
    # ------------------------------------------------------------------ #

    async def get_trades(
        self,
        market: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        start: Optional[int] = None,  # unix timestamp
        end: Optional[int] = None,
    ) -> list[dict]:
        """Return recent trades, optionally filtered by market condition_id."""
        params: dict = {"limit": limit, "offset": offset}
        if market:
            params["market"] = market
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        data = await self._get(DATA_BASE, "/trades", params)
        return data if isinstance(data, list) else data.get("trades", [])

    async def get_wallet_activity(
        self,
        wallet: str,
        limit: int = 50,
    ) -> list[dict]:
        """Return recent on-chain activity for a wallet address."""
        data = await self._get(DATA_BASE, "/activity", {
            "user": wallet,
            "limit": limit,
        })
        return data if isinstance(data, list) else []

    async def get_wallet_trade_count(self, wallet: str) -> int:
        """Estimate the total number of trades a wallet has made."""
        activity = await self.get_wallet_activity(wallet, limit=500)
        return sum(1 for a in activity if a.get("type") == "TRADE")

    async def get_wallet_first_trade_timestamp(self, wallet: str) -> Optional[int]:
        """Return the unix timestamp of the wallet's oldest known trade, or None."""
        data = await self._get(DATA_BASE, "/activity", {
            "user": wallet,
            "limit": 500,
            "sortBy": "TIMESTAMP",
            "sortDirection": "ASC",
            "type": "TRADE",
        })
        items = data if isinstance(data, list) else []
        if items:
            return items[0].get("timestamp")
        return None

    # ------------------------------------------------------------------ #
    # CLOB API — prices & orderbook
    # ------------------------------------------------------------------ #

    async def get_midpoint(self, token_id: str) -> Optional[float]:
        """Return the current midpoint price (0–1) for an outcome token."""
        data = await self._get(CLOB_BASE, f"/midpoint?token_id={token_id}")
        mid = data.get("mid")
        return float(mid) if mid is not None else None

    async def get_orderbook(self, token_id: str) -> dict:
        """Return the current orderbook for an outcome token."""
        return await self._get(CLOB_BASE, f"/book?token_id={token_id}")

    async def get_price_history(
        self,
        market_id: str,
        interval: str = "1h",     # 1m 5m 1h 6h 1d
        fidelity: int = 60,       # data points
    ) -> list[dict]:
        """Return OHLCV-style price history for a market."""
        data = await self._get(CLOB_BASE, "/prices-history", {
            "market": market_id,
            "interval": interval,
            "fidelity": fidelity,
        })
        return data.get("history", [])

    async def get_last_trade_price(self, token_id: str) -> Optional[float]:
        """Return the last traded price for a token."""
        data = await self._get(CLOB_BASE, f"/last-trade-price?token_id={token_id}")
        p = data.get("price")
        return float(p) if p is not None else None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def market_end_timestamp(market: dict) -> Optional[int]:
        """Parse the market end date into a unix timestamp."""
        end_str = market.get("endDate") or market.get("end_date_iso")
        if not end_str:
            return None
        try:
            dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except ValueError:
            return None

    @staticmethod
    def hours_until_close(market: dict) -> Optional[float]:
        end_ts = PolymarketClient.market_end_timestamp(market)
        if end_ts is None:
            return None
        return (end_ts - time.time()) / 3600
