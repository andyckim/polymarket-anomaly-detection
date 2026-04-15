"""
backtest.py — Replay poly_data CSV exports through AnomalyDetector scoring.

Usage:
    python backtest.py --trades path/to/processed/trades.csv \
                       --markets path/to/markets.csv

    # with a custom detection profile:
    python backtest.py --trades ... --markets ... --config insider_whale.json

    # save alert rows to CSV:
    python backtest.py --trades ... --markets ... --out results.csv

    # suppress per-alert output, show only the summary:
    python backtest.py --trades ... --markets ... --quiet

poly_data column reference
    trades.csv : timestamp, market_id (token ID), maker, taker,
                 nonusdc_side, maker_direction, taker_direction,
                 price, usd_amount, token_amount, transactionHash
    markets.csv: createdAt, id, question, answer1, answer2, neg_risk,
                 market_slug, token1, token2, condition_id, volume,
                 ticker, closedTime
"""

import asyncio
import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

from bot import load_config
from detector import AnomalyDetector, AnomalyResult


log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# BacktestClient — stands in for PolymarketClient, no network calls
# ------------------------------------------------------------------ #

class BacktestClient:
    """
    Drop-in replacement for PolymarketClient during backtesting.

    Wallet stats (first-trade timestamp, total trade count) are
    pre-computed from the full trades CSV so every lookup is O(1)
    and fully deterministic.

    hours_until_close() is computed against the current trade's
    timestamp rather than wall-clock time so timing scores reflect
    historical market state accurately.
    """

    def __init__(
        self,
        wallet_first_ts: dict[str, float],
        wallet_trade_count: dict[str, int],
        current_ts_getter,          # callable() -> float
    ):
        self._first_ts = wallet_first_ts
        self._trade_count = wallet_trade_count
        self._current_ts = current_ts_getter

    async def get_wallet_first_trade_timestamp(self, wallet: str) -> Optional[float]:
        return self._first_ts.get(wallet)

    async def get_wallet_trade_count(self, wallet: str) -> int:
        return self._trade_count.get(wallet, 0)

    def hours_until_close(self, market: dict) -> Optional[float]:
        close_str = market.get("closedTime") or market.get("endDate")
        if not close_str:
            return None
        try:
            dt = datetime.fromisoformat(str(close_str).replace("Z", "+00:00"))
            return (dt.timestamp() - self._current_ts()) / 3600
        except (ValueError, TypeError):
            return None


# ------------------------------------------------------------------ #
# CSV loaders
# ------------------------------------------------------------------ #

def load_trades(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: r.get("timestamp", ""))
    return rows


def load_markets(path: str) -> dict[str, dict]:
    """
    Returns token_id -> market_row for every token referenced in
    the markets CSV (both token1 and token2 columns).
    """
    token_to_market: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for col in ("token1", "token2"):
                tid = row.get(col, "").strip()
                if tid:
                    token_to_market[tid] = row
    return token_to_market


# ------------------------------------------------------------------ #
# Pre-compute wallet stats
# ------------------------------------------------------------------ #

def build_wallet_stats(
    trades: list[dict],
) -> tuple[dict[str, float], dict[str, int]]:
    """
    Single-pass over all trades to build:
      first_ts[wallet]  = earliest trade timestamp (float, Unix seconds)
      count[wallet]     = total number of trades seen
    Both maker and taker are treated as participants in each trade.
    """
    first_ts: dict[str, float] = {}
    count: dict[str, int] = {}

    for row in trades:
        ts_str = row.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError):
            continue

        for field in ("maker", "taker"):
            wallet = row.get(field, "").strip()
            if not wallet:
                continue
            if wallet not in first_ts or ts < first_ts[wallet]:
                first_ts[wallet] = ts
            count[wallet] = count.get(wallet, 0) + 1

    return first_ts, count


# ------------------------------------------------------------------ #
# Trade normalisation
# ------------------------------------------------------------------ #

def normalise_trade(row: dict, market: dict) -> dict:
    """
    Convert a poly_data trades row into the trade dict format that
    AnomalyDetector.score_trade() expects.

    The taker is the aggressor (active side), so we use taker as the
    wallet and taker_direction as the side.  Falls back to maker if
    taker is absent.
    """
    wallet = row.get("taker", "").strip() or row.get("maker", "").strip()
    side = (row.get("taker_direction") or row.get("maker_direction") or "").upper()

    return {
        "usdcSize":        row.get("usd_amount", "0"),
        "price":           row.get("price", "0"),
        "side":            side,
        "conditionId":     market.get("condition_id", ""),
        "proxyWallet":     wallet,
        "transactionHash": row.get("transactionHash", ""),
        "_token_id":       row.get("market_id", ""),   # poly_data market_id is the token ID
    }


def normalise_market(row: dict) -> dict:
    """
    Add field aliases so the detector can find open/close times
    regardless of which column naming convention it expects.
    """
    m = dict(row)
    # Detector's _score_timing looks for 'startDate' or 'start_date_iso'
    if "createdAt" in m and "startDate" not in m:
        m["startDate"] = m["createdAt"]
    return m


# ------------------------------------------------------------------ #
# Backtest runner
# ------------------------------------------------------------------ #

async def run_backtest(
    trades: list[dict],
    token_to_market: dict[str, dict],
    config_path: Optional[str],
    quiet: bool = False,
) -> tuple[list[dict], int, int]:
    """
    Replay every trade through AnomalyDetector and return
    (alert_rows, skipped_count, total_count).
    """
    cfg = load_config(config_path)
    if not quiet:
        print(cfg.summary())
        print()

    wallet_first_ts, wallet_count = build_wallet_stats(trades)

    # Shared mutable timestamp so BacktestClient stays in sync with
    # the trade being scored without needing a mutable closure variable
    current_ts: list[float] = [0.0]

    client = BacktestClient(
        wallet_first_ts=wallet_first_ts,
        wallet_trade_count=wallet_count,
        current_ts_getter=lambda: current_ts[0],
    )
    detector = AnomalyDetector(cfg, client)

    # Patch detector.time once for the whole run so that both
    # AnomalyDetector and RollingVolumeTracker use the replayed
    # timestamp instead of wall-clock time.
    mock_time = MagicMock()
    mock_time.time.return_value = 0.0

    results: list[dict] = []
    skipped = 0
    seen_tx: set[str] = set()
    last_price: dict[str, float] = {}   # token_id -> most-recent price

    total = len(trades)
    severity_icons = {"HIGH": "\U0001f534", "MEDIUM": "\U0001f7e0", "LOW": "\U0001f7e1"}

    with patch("detector.time", mock_time):
        for i, row in enumerate(trades):
            if not quiet and i % 5_000 == 0:
                print(f"  Progress: {i:,}/{total:,} …   ", end="\r", flush=True)

            # Parse timestamp
            ts_str = row.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError):
                skipped += 1
                continue

            # Resolve market
            token_id = row.get("market_id", "").strip()
            raw_market = token_to_market.get(token_id)
            if not raw_market:
                skipped += 1
                continue

            market = normalise_market(raw_market)
            trade  = normalise_trade(row, market)

            # Deduplicate by transaction hash
            tx = trade.get("transactionHash", "")
            if tx and tx in seen_tx:
                continue
            if tx:
                seen_tx.add(tx)

            # Maintain pre/post price cache
            pre_price = last_price.get(token_id)
            try:
                post_price = float(trade.get("price") or 0)
            except (ValueError, TypeError):
                post_price = 0.0
            if token_id and post_price:
                last_price[token_id] = post_price

            # Advance the mock clock to this trade's timestamp
            current_ts[0] = ts
            mock_time.time.return_value = ts

            result = await detector.score_trade(trade, market, pre_price)

            if result:
                if not quiet:
                    print()  # newline after progress
                    icon = severity_icons.get(result.severity, "\u26aa")
                    print(f"\n{icon} {result.summary()}\n{'─' * 60}")

                results.append({
                    "timestamp":    ts_str,
                    "market_slug":  market.get("market_slug", ""),
                    "question":     market.get("question", ""),
                    "condition_id": market.get("condition_id", ""),
                    "wallet":       trade.get("proxyWallet", ""),
                    "side":         trade.get("side", ""),
                    "usd_amount":   trade.get("usdcSize", ""),
                    "price":        trade.get("price", ""),
                    "total_score":  result.total_score,
                    "severity":     result.severity,
                    "factors":      "; ".join(
                        f"{f.name}={f.score:.2f}"
                        for f in result.factors if f.score > 0
                    ),
                })

    if not quiet:
        print()  # clear progress line

    return results, skipped, total


# ------------------------------------------------------------------ #
# Summary
# ------------------------------------------------------------------ #

def print_summary(results: list[dict], total: int, skipped: int) -> None:
    n_alerts  = len(results)
    n_scored  = total - skipped
    alert_pct = f"{n_alerts / n_scored * 100:.3f}%" if n_scored else "n/a"

    print("\n" + "=" * 62)
    print("BACKTEST SUMMARY")
    print("=" * 62)
    print(f"  Trades processed : {n_scored:>10,}")
    print(f"  Trades skipped   : {skipped:>10,}  (no market match / bad ts)")
    print(f"  Alerts fired     : {n_alerts:>10,}  ({alert_pct} of scored)")

    if results:
        sev: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for r in results:
            sev[r["severity"]] = sev.get(r["severity"], 0) + 1
        print(f"    HIGH           : {sev['HIGH']:>10,}")
        print(f"    MEDIUM         : {sev['MEDIUM']:>10,}")
        print(f"    LOW            : {sev['LOW']:>10,}")

        top = sorted(results, key=lambda r: float(r["total_score"]), reverse=True)[:10]
        print("\nTop anomalous trades by score:")
        hdr = f"  {'Score':>6}  {'Sev':6}  {'Wallet':42}  {'Market'}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in top:
            wallet   = (r["wallet"] or "")[:42]
            question = (r.get("question") or r.get("market_slug") or "")[:38]
            print(f"  {float(r['total_score']):>6.2f}  {r['severity']:6}  {wallet:42}  {question}")

    print("=" * 62)


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest AnomalyDetector thresholds against "
            "poly_data CSV exports."
        )
    )
    parser.add_argument(
        "--trades",  required=True,
        help="Path to poly_data processed/trades.csv",
    )
    parser.add_argument(
        "--markets", required=True,
        help="Path to poly_data markets.csv",
    )
    parser.add_argument(
        "--config",  default=None,
        help="Path to JSON detection profile (e.g. insider_whale.json)",
    )
    parser.add_argument(
        "--out",     default=None,
        help="Write alert rows to this CSV file",
    )
    parser.add_argument(
        "--quiet",   action="store_true",
        help="Suppress per-alert output; show only the final summary",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,      # hide detector/client noise during replay
        format="%(levelname)s %(message)s",
    )

    print(f"Loading trades  : {args.trades}")
    trades = load_trades(args.trades)
    print(f"  {len(trades):,} rows")

    print(f"Loading markets : {args.markets}")
    token_to_market = load_markets(args.markets)
    print(f"  {len(token_to_market):,} token→market mappings\n")

    results, skipped, total = asyncio.run(
        run_backtest(trades, token_to_market, args.config, quiet=args.quiet)
    )

    print_summary(results, total, skipped)

    if args.out:
        if not results:
            print("\nNo alerts to write.")
        else:
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
            print(f"\nResults written to {args.out}  ({len(results):,} rows)")


if __name__ == "__main__":
    main()
