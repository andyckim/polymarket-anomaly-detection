import asyncio
import json
from typing import Awaitable, Callable, Optional

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from polymarket_client import PolymarketClient


ProbabilityEstimator = Callable[[dict], Awaitable[float]]


class _Estimate(BaseModel):
    reasoning: str = Field(description="2-3 sentences justifying the probability.")
    probability: float = Field(ge=0.0, le=1.0)


_SYSTEM_PROMPT = """You are a calibrated forecaster for binary prediction markets.

Given a market question, return your independent probability (0.0 to 1.0) that it resolves YES.

Rules:
- Do NOT anchor on any implied market price. Estimate the true probability from first principles.
- Use base rates, known facts, and reasoning about the event's mechanics.
- When genuinely uncertain, stay near 0.5 rather than fabricating confidence.
- Keep reasoning to 2-3 sentences."""


def claude_estimator(anthropic_client: AsyncAnthropic) -> ProbabilityEstimator:
    """Return an estimator that asks Claude for a calibrated probability per market."""
    async def _estimate(market: dict) -> float:
        question = market.get("question", "")
        description = market.get("description", "")
        end_date = market.get("endDate", "")
        user = f"Question: {question}\nEnd date: {end_date}\n\nDescription: {description}"

        response = await anthropic_client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
            output_format=_Estimate,
        )
        return response.parsed_output.probability

    return _estimate


async def scan_and_queue(
    client: PolymarketClient,
    anthropic_client: AsyncAnthropic,
    *,
    min_gap: float = 0.07,
    min_depth_usd: float = 500.0,
    min_hours: float = 4.0,
    max_hours: float = 48.0,
    output_path: str = "queue.json",
) -> list[dict]:
    """Pull active markets, score each on gap/depth/hours, save survivors."""
    estimate_probability = claude_estimator(anthropic_client)
    markets = await client.get_all_active_markets()
    survivors: list[dict] = []

    for market in markets:
        hours = PolymarketClient.hours_until_close(market)
        if hours is None or hours < min_hours or hours > max_hours:
            continue

        yes_token = _yes_token_id(market)
        if not yes_token:
            continue

        price = await client.get_midpoint(yes_token)
        if price is None:
            continue

        book = await client.get_orderbook(yes_token)
        bid_depth = _side_depth_usd(book.get("bids", []))
        ask_depth = _side_depth_usd(book.get("asks", []))
        if min(bid_depth, ask_depth) < min_depth_usd:
            continue

        estimate = await estimate_probability(market)
        gap = abs(estimate - price)
        if gap < min_gap:
            continue

        survivors.append({
            "slug":        market.get("slug"),
            "question":    market.get("question"),
            "conditionId": market.get("conditionId"),
            "yes_token":   yes_token,
            "price":       round(price, 4),
            "estimate":    round(estimate, 4),
            "gap":         round(gap, 4),
            "bid_depth":   round(bid_depth, 2),
            "ask_depth":   round(ask_depth, 2),
            "hours":       round(hours, 2),
            "ev":          round(gap * min(bid_depth, ask_depth), 2),
        })

    survivors.sort(key=lambda s: s["ev"], reverse=True)

    with open(output_path, "w") as f:
        json.dump(survivors, f, indent=2)

    print(f"scanned {len(markets)} > {len(survivors)} survivors — saved to {output_path}")
    return survivors


def _yes_token_id(market: dict) -> Optional[str]:
    raw = market.get("clobTokenIds")
    if not raw:
        return None
    try:
        ids = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    return ids[0] if ids else None


def _side_depth_usd(levels: list[dict]) -> float:
    total = 0.0
    for lvl in levels:
        try:
            total += float(lvl.get("price", 0)) * float(lvl.get("size", 0))
        except (TypeError, ValueError):
            continue
    return total


if __name__ == "__main__":
    async def _main():
        anthropic_client = AsyncAnthropic()
        async with PolymarketClient() as client:
            await scan_and_queue(client, anthropic_client)

    asyncio.run(_main())
