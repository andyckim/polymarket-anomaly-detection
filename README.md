# Polymarket Anomaly Bot

Polls Polymarket's public APIs and scores recent trades for anomalous behaviour.
No API key required — all data endpoints used are public.

## Files

| File | Purpose |
|---|---|
| `config.py` | All detection thresholds in one place — edit here |
| `detector.py` | Scoring logic for each anomaly factor |
| `polymarket_client.py` | Async HTTP wrapper for Polymarket APIs |
| `bot.py` | Polling loop, alert handler, CLI entry point |
| `config_example.json` | JSON override file for runtime config |

## Install

```bash
pip install aiohttp
```

## Run

```bash
# Default config, scan all markets every 30s
python bot.py

# Custom sensitivity overrides from JSON
python bot.py --config config_example.json

# Watch specific markets only, poll every 60s
python bot.py --interval 60 --markets will-trump-win-2024 will-fed-cut-rates-in-june
```

## Anomaly Factors

Each factor produces an independent score. Scores are summed and compared
against `scoring.alert_threshold` (default **3.0**).

| Factor | Config class | What it measures |
|---|---|---|
| **Trade size** | `TradeSizeConfig` | Absolute USDC size — large / whale tiers |
| **Account** | `AccountConfig` | Wallet age and number of prior trades |
| **Volume dominance** | `VolumeConfig` | Trade as % of 24h market volume |
| **Volume spike** | `VolumeConfig` | Rolling-window volume vs. baseline |
| **Price impact** | `PriceImpactConfig` | Price move caused by the trade |
| **Timing** | `TimingConfig` | Late-close or early-open anomalies |

## Tuning Guide

**Too many alerts?** Raise `scoring.alert_threshold` (try 4–5) or raise
`trade_size.min_trade_size_usd` to ignore small trades.

**Missing real signals?** Lower `scoring.alert_threshold` or tighten individual
factor thresholds (e.g. lower `volume.trade_volume_fraction_threshold` to 0.05).

**Noisy new-account flag?** Raise `account.new_account_age_days` or lower
`account.new_account_score_multiplier` toward 1.0.

**Want to focus on thin markets?** Lower `volume.thin_market_volume_usd` and
raise `volume.thin_market_score_multiplier`.

## Sending Alerts

Replace the `alert()` function in `bot.py` with your preferred sink:

```python
# Slack example
import httpx

def alert(result: AnomalyResult):
    httpx.post(SLACK_WEBHOOK, json={"text": result.summary()})
```
