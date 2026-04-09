"""
bot.py — Polymarket anomaly detection bot.

Usage:
    python bot.py                   # run with default config
    python bot.py --config my.json  # load config overrides from JSON

The bot polls Polymarket's public APIs on a configurable interval,
scores recent trades through AnomalyDetector, and prints alerts.
Swap out the alert() method to send to Slack, Discord, email, etc.
"""

import asyncio
import json
import logging
import argparse
import time
from dataclasses import asdict

import aiohttp

from config import AnomalyConfig, AccountConfig, VolumeConfig, TradeSizeConfig
from config import PriceImpactConfig, TimingConfig, ScoringConfig
from detector import AnomalyDetector, AnomalyResult
from polymarket_client import PolymarketClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Alert handler — replace this with your notification sink
# ------------------------------------------------------------------ #

def alert(result: AnomalyResult):
    """Called whenever a trade exceeds the anomaly threshold."""
    severity_colors = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}
    icon = severity_colors.get(result.severity, "⚪")
    print(f"\n{icon} {result.summary()}\n{'─' * 60}")
    # TODO: post to Slack / Discord / email / database


# ------------------------------------------------------------------ #
# Config loader
# ------------------------------------------------------------------ #

def load_config(path: str | None) -> AnomalyConfig:
    cfg = AnomalyConfig()
    if not path:
        return cfg
    with open(path) as f:
        overrides = json.load(f)

    sub_map = {
        "account": (cfg.account, AccountConfig),
        "volume": (cfg.volume, VolumeConfig),
        "trade_size": (cfg.trade_size, TradeSizeConfig),
        "price_impact": (cfg.price_impact, PriceImpactConfig),
        "timing": (cfg.timing, TimingConfig),
        "scoring": (cfg.scoring, ScoringConfig),
    }
    for key, value in overrides.items():
        if key in sub_map:
            obj, _ = sub_map[key]
            for field, v in value.items():
                if hasattr(obj, field):
                    setattr(obj, field, v)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)

    return cfg


# ------------------------------------------------------------------ #
# Bot
# ------------------------------------------------------------------ #

class AnomalyBot:
    def __init__(
        self,
        config: AnomalyConfig,
        poll_interval_seconds: float = 30.0,
        trades_per_market: int = 50,
    ):
        self.config = config
        self.poll_interval = poll_interval_seconds
        self.trades_per_market = trades_per_market
        self._seen_tx: set[str] = set()    # deduplicate already-scored trades
        self._market_cache: dict[str, dict] = {}
        self._last_price: dict[str, float] = {}  # token_id -> last known price

    async def run(self):
        log.info("Starting Polymarket Anomaly Bot")
        log.info(self.config.summary())

        async with aiohttp.ClientSession() as session:
            client = PolymarketClient(session)
            detector = AnomalyDetector(self.config, client)

            while True:
                try:
                    await self._poll(client, detector)
                except Exception as e:
                    log.error("Poll error: %s", e, exc_info=True)
                await asyncio.sleep(self.poll_interval)

    async def _poll(self, client: PolymarketClient, detector: AnomalyDetector):
        # Refresh market list periodically
        if not self._market_cache or int(time.time()) % 300 < self.poll_interval:
            await self._refresh_markets(client)

        markets = list(self._market_cache.values())
        if self.config.watched_markets:
            markets = [m for m in markets
                       if m.get("conditionId") in self.config.watched_markets
                       or m.get("slug") in self.config.watched_markets]

        log.info("Scanning %d markets …", len(markets))

        tasks = [self._scan_market(client, detector, m) for m in markets]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Keep seen-tx set bounded
        if len(self._seen_tx) > 100_000:
            self._seen_tx.clear()

    async def _refresh_markets(self, client: PolymarketClient):
        try:
            markets = await client.get_all_active_markets()
            ignored = set(self.config.ignored_markets)
            self._market_cache = {
                m["conditionId"]: m
                for m in markets
                if m.get("conditionId")
                and m.get("conditionId") not in ignored
                and m.get("slug") not in ignored
            }
            log.info("Market cache refreshed: %d markets", len(self._market_cache))
        except Exception as e:
            log.warning("Failed to refresh market cache: %s", e)

    async def _scan_market(
        self,
        client: PolymarketClient,
        detector: AnomalyDetector,
        market: dict,
    ):
        condition_id = market.get("conditionId", "")
        try:
            trades = await client.get_trades(
                market=condition_id,
                limit=self.trades_per_market,
            )
        except Exception as e:
            log.debug("Failed to fetch trades for %s: %s", condition_id, e)
            return

        # Get token id for pre-trade price lookup (YES token)
        tokens = market.get("tokens") or []
        yes_token_id = next(
            (t.get("token_id") for t in tokens if t.get("outcome", "").lower() == "yes"),
            None,
        )

        for trade in trades:
            tx_hash = trade.get("transactionHash") or trade.get("id")
            if tx_hash and tx_hash in self._seen_tx:
                continue
            if tx_hash:
                self._seen_tx.add(tx_hash)

            # Look up pre-trade price from our cache
            pre_price = self._last_price.get(yes_token_id) if yes_token_id else None
            post_price = float(trade.get("price") or 0)
            if yes_token_id and post_price:
                self._last_price[yes_token_id] = post_price

            result = await detector.score_trade(trade, market, pre_price)
            if result:
                alert(result)


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

async def main():
    parser = argparse.ArgumentParser(description="Polymarket Anomaly Bot")
    parser.add_argument("--config", help="Path to JSON config overrides", default=None)
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Poll interval in seconds (default: 30)")
    parser.add_argument("--markets", nargs="*",
                        help="condition_ids or slugs to watch (default: all)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.markets:
        cfg.watched_markets = args.markets

    bot = AnomalyBot(cfg, poll_interval_seconds=args.interval)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
