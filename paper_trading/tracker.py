"""Live paper trading tracker.

Processes today's trend + sentiment + risk signals and updates simulated
paper positions stored in paper_trades.db.

Designed to be called daily (after run_daily_risk.py completes), either:
  - standalone: python -m trend_following.paper_trading.tracker
  - via CLI:    python cli.py paper
  - via runner: run_paper_trading.py (at trading root)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_TRADING_ROOT = Path(__file__).resolve().parents[2]
_RISK_BACKEND = _TRADING_ROOT / "risk_calculator" / "backend"
if str(_RISK_BACKEND) not in sys.path:
    sys.path.insert(0, str(_RISK_BACKEND))

from ..config import TrendFollowingConfig
from ..indicators.volatility import ATR
from ..signals.generator import generate_signal
from ..position_sizing.sizer import shares_to_buy
from . import db as paper_db

log = logging.getLogger(__name__)

TICKER_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc.",
    "NVDA": "NVIDIA Corporation",
    "JPM": "JPMorgan Chase & Co.",
}


def _fetch_ohlc(ticker: str, lookback_days: int):
    """Load OHLCV data via the risk_calculator market_data service."""
    from app.services.market_data import fetch_ohlcv
    return fetch_ohlcv(ticker, lookback_days)


def run_paper_trading(
    cfg: Optional[TrendFollowingConfig] = None,
    force: bool = False,
) -> List[Dict]:
    """Process today's signals and update paper positions for all tickers.

    Args:
        cfg:   Strategy configuration (uses defaults if None).
        force: If True, skip the already-ran-today guard and run regardless.

    Returns:
        List of action dicts: one per ticker, with signal, action taken, and P&L.
    """
    if cfg is None:
        cfg = TrendFollowingConfig()

    paper_db.init_db()

    if not force and paper_db.has_run_today():
        log.warning(
            "Paper trading already ran today — skipping. Pass force=True to override."
        )
        return []
    actions = []

    # ── Compute current portfolio value (cash + cost basis of open positions) ─
    _positions_snapshot = paper_db.get_positions()
    _cash = paper_db.get_cash_balance(cfg.backtest.initial_capital)
    _invested = sum(p["shares"] * p["avg_cost"] for p in _positions_snapshot)
    portfolio_value = _cash + _invested
    if portfolio_value <= 0:
        portfolio_value = cfg.backtest.initial_capital
    log.info(
        "Portfolio value: $%.2f  (cash=$%.2f  invested=$%.2f)",
        portfolio_value, _cash, _invested,
    )

    for ticker in cfg.tickers:
        log.info("-- %s --", ticker)

        try:
            ohlc = _fetch_ohlc(ticker, cfg.lookback_days)
        except Exception as exc:
            log.warning("  No price data for %s: %s — skipping", ticker, exc)
            actions.append(
                {
                    "ticker": ticker,
                    "signal": "ERROR",
                    "action_taken": "SKIP",
                    "shares": 0,
                    "price": None,
                    "pnl": None,
                    "reason": str(exc),
                }
            )
            continue

        current_price = float(ohlc["Close"].iloc[-1])
        positions = {p["ticker"]: p for p in paper_db.get_positions()}
        pos = positions.get(ticker.upper(), {})
        held = pos.get("shares", 0)
        avg_cost = pos.get("avg_cost", 0.0)
        stored_stop = pos.get("atr_stop")

        action_taken = "HOLD"
        shares_traded = 0
        pnl = None

        # ── ATR stop check on open positions ──────────────────────────────────
        if cfg.atr_stop.enabled and held > 0 and stored_stop is not None:
            if current_price <= stored_stop:
                gross_pnl = (current_price - avg_cost) * held
                commission = current_price * held * cfg.backtest.commission_pct
                net_pnl = gross_pnl - commission
                paper_db.log_trade(
                    ticker=ticker,
                    action="SELL",
                    shares=held,
                    price=current_price,
                    commission=commission,
                    pnl=net_pnl,
                    reason="ATR_STOP_HIT",
                    signal_strength=0.0,
                )
                paper_db.upsert_position(ticker, 0, 0.0)
                action_taken = "STOP_SELL"
                shares_traded = held
                pnl = net_pnl
                log.info(
                    "  ATR STOP HIT: sold %d shares @ $%.2f (stop=%.2f) | Net P&L: $%.2f",
                    held, current_price, stored_stop, net_pnl,
                )
                actions.append(
                    {
                        "ticker": ticker,
                        "signal": "STOP",
                        "action_taken": action_taken,
                        "shares": shares_traded,
                        "price": current_price,
                        "pnl": pnl,
                        "reason": "ATR_STOP_HIT",
                        "sentiment": None,
                        "risk_score": None,
                    }
                )
                continue

            # Trailing profit stop — exit if price falls N×ATR from peak since entry
            if cfg.atr_stop.profit_stop_enabled:
                atr_ind = ATR(period=cfg.atr_stop.period)
                atr_val = atr_ind.latest_atr(ohlc)
                if atr_val > 0:
                    # Peak = max of stored peak (or entry price) and current price
                    stored_peak = pos.get("peak_price", avg_cost)
                    peak = max(stored_peak, current_price)
                    profit_stop = peak - cfg.atr_stop.profit_stop_atr_mult * atr_val
                    if current_price <= profit_stop:
                        gross_pnl = (current_price - avg_cost) * held
                        commission = current_price * held * cfg.backtest.commission_pct
                        net_pnl = gross_pnl - commission
                        paper_db.log_trade(
                            ticker=ticker, action="SELL", shares=held, price=current_price,
                            commission=commission, pnl=net_pnl, reason="PROFIT_STOP_HIT",
                            signal_strength=0.0,
                        )
                        paper_db.upsert_position(ticker, 0, 0.0)
                        action_taken = "PROFIT_SELL"
                        shares_traded = held
                        pnl = net_pnl
                        log.info(
                            "  PROFIT STOP HIT: sold %d @ $%.2f (peak=%.2f, stop=%.2f) | P&L: $%.2f",
                            held, current_price, peak, profit_stop, net_pnl,
                        )
                        actions.append(
                            {"ticker": ticker, "signal": "STOP", "action_taken": action_taken,
                             "shares": shares_traded, "price": current_price, "pnl": pnl,
                             "reason": "PROFIT_STOP_HIT", "sentiment": None, "risk_score": None})
                        continue
                    # Update stored peak
                    if peak > stored_peak:
                        paper_db.update_peak_price(ticker, peak)

            # Trail: ratchet stop up as price rises — never moves down
            if cfg.atr_stop.trail:
                atr_ind = ATR(period=cfg.atr_stop.period)
                atr_val = atr_ind.latest_atr(ohlc)
                if atr_val > 0:
                    candidate = current_price - cfg.atr_stop.multiplier * atr_val
                    if candidate > stored_stop:
                        paper_db.update_atr_stop(ticker, candidate)
                        log.info(
                            "  ATR TRAIL updated: %.2f → %.2f", stored_stop, candidate
                        )

        signal = generate_signal(ticker, ohlc, cfg)
        log.info(
            "  Signal: %s (dir=%+.0f, strength=%.2f) | %s",
            signal.action,
            signal.trend_direction,
            signal.filtered_strength,
            signal.reason,
        )

        if signal.action == "BUY" and held == 0:
            n_shares = shares_to_buy(
                signal,
                portfolio_value,
                current_price,
                cfg,
            )
            if n_shares > 0:
                commission = current_price * n_shares * cfg.backtest.commission_pct
                atr_stop_price: Optional[float] = None
                if cfg.atr_stop.enabled:
                    atr_ind = ATR(period=cfg.atr_stop.period)
                    atr_stop_price = atr_ind.stop_price(
                        ohlc, current_price, cfg.atr_stop.multiplier, "long"
                    )
                paper_db.log_trade(
                    ticker=ticker,
                    action="BUY",
                    shares=n_shares,
                    price=current_price,
                    commission=commission,
                    reason=signal.reason,
                    sentiment=signal.sentiment,
                    risk_score=signal.risk_score,
                    signal_strength=signal.filtered_strength,
                )
                paper_db.upsert_position(ticker, n_shares, current_price, atr_stop=atr_stop_price)
                action_taken = "BUY"
                shares_traded = n_shares
                log.info(
                    "  PAPER BUY: %d shares @ $%.2f (stop=%.2f)",
                    n_shares, current_price, atr_stop_price or 0.0,
                )

        elif signal.action == "SELL" and held > 0:
            gross_pnl = (current_price - avg_cost) * held
            commission = current_price * held * cfg.backtest.commission_pct
            net_pnl = gross_pnl - commission
            paper_db.log_trade(
                ticker=ticker,
                action="SELL",
                shares=held,
                price=current_price,
                commission=commission,
                pnl=net_pnl,
                reason=signal.reason,
                sentiment=signal.sentiment,
                risk_score=signal.risk_score,
                signal_strength=signal.filtered_strength,
            )
            paper_db.upsert_position(ticker, 0, 0.0)
            action_taken = "SELL"
            shares_traded = held
            pnl = net_pnl
            log.info(
                "  PAPER SELL: %d shares @ $%.2f | Net P&L: $%.2f",
                held, current_price, net_pnl,
            )

        actions.append(
            {
                "ticker": ticker,
                "signal": signal.action,
                "action_taken": action_taken,
                "shares": shares_traded,
                "price": current_price,
                "pnl": pnl,
                "reason": signal.reason,
                "sentiment": signal.sentiment,
                "risk_score": signal.risk_score,
            }
        )

    paper_db.record_daily_run(tickers_processed=len(cfg.tickers))
    return actions
