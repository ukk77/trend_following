"""Sentiment-weighted position sizing.

Position size = base_position_pct × sentiment_multiplier, capped at max_position_pct.

Sentiment alignment rules:
    positive sentiment + BUY → full multiplier (sentiment_agree_mult, default 1.0)
    neutral  sentiment + BUY → reduced size   (sentiment_neutral_mult, default 0.5)
    negative sentiment + BUY → skip           (sentiment_disagree_mult, default 0.0)

If the risk calculator has a valid Kelly fraction for this ticker, that is
used as a floor/ceiling check (optional, controlled by use_kelly_fraction).
"""
from __future__ import annotations

from typing import Optional

from ..config import TrendFollowingConfig
from ..signals.generator import Signal


def compute_position_dollars(
    signal: Signal,
    portfolio_value: float,
    cfg: TrendFollowingConfig,
    kelly_fraction: Optional[float] = None,
) -> float:
    """Return the position size in dollars for the given signal.

    Args:
        signal: The generated trading signal.
        portfolio_value: Current total portfolio value in dollars.
        cfg: Strategy configuration.
        kelly_fraction: Optional Kelly fraction from risk calculator.

    Returns:
        Dollar amount to allocate (0.0 for HOLD or blocked trades).
    """
    if signal.action == "HOLD":
        return 0.0

    ps = cfg.position_sizing
    base_usd = portfolio_value * (ps.base_position_pct / 100.0)
    max_usd = portfolio_value * (ps.max_position_pct / 100.0)

    if signal.action == "BUY":
        sentiment = signal.sentiment
        conf = signal.sentiment_confidence or 0.0

        if sentiment == "positive" and conf >= cfg.signal.min_sentiment_confidence:
            multiplier = ps.sentiment_agree_mult
        elif sentiment == "negative":
            multiplier = ps.sentiment_disagree_mult
        else:
            multiplier = ps.sentiment_neutral_mult

        size = base_usd * multiplier

        # Optionally cap against Kelly fraction
        if ps.use_kelly_fraction and kelly_fraction is not None and kelly_fraction > 0:
            kelly_capped = min(kelly_fraction, ps.kelly_cap)
            kelly_usd = portfolio_value * kelly_capped
            size = min(size, kelly_usd)

    else:
        # SELL — use full base size (closing logic; actual shares come from held position)
        size = base_usd

    return float(min(size, max_usd))


def shares_to_buy(
    signal: Signal,
    portfolio_value: float,
    current_price: float,
    cfg: TrendFollowingConfig,
    kelly_fraction: Optional[float] = None,
) -> int:
    """Return the number of whole shares to buy/sell.

    Args:
        signal: The trading signal.
        portfolio_value: Current total portfolio value in dollars.
        current_price: Current price per share.
        cfg: Strategy configuration.
        kelly_fraction: Optional Kelly fraction from risk calculator.

    Returns:
        Number of whole shares (>= 0).
    """
    if current_price <= 0:
        return 0
    dollar_size = compute_position_dollars(signal, portfolio_value, cfg, kelly_fraction)
    return max(0, int(dollar_size / current_price))
