"""Simulated portfolio — tracks cash, positions, trades, and equity curve.

All prices are assumed to be adjusted-close (auto_adjust=True from yfinance).
Slippage is applied externally before calling buy/sell.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Trade:
    """Record of a single simulated trade."""
    date: str
    ticker: str
    action: str
    shares: int
    price: float
    commission: float
    pnl: Optional[float] = None


class Portfolio:
    """Cash + equity portfolio for backtesting.

    Tracks:
        - cash balance
        - open positions (shares per ticker)
        - average cost basis per ticker
        - trade log
        - daily equity snapshots
    """

    def __init__(self, initial_capital: float, commission_pct: float = 0.001) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_pct = commission_pct

        self._positions: Dict[str, int] = {}
        self._avg_cost: Dict[str, float] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

    # ── Queries ───────────────────────────────────────────────────────────────

    def shares_held(self, ticker: str) -> int:
        return self._positions.get(ticker, 0)

    def cost_basis(self, ticker: str) -> float:
        return self._avg_cost.get(ticker, 0.0)

    def holdings_value(self, prices: Dict[str, float]) -> float:
        return sum(self._positions.get(t, 0) * p for t, p in prices.items())

    def equity(self, prices: Dict[str, float]) -> float:
        return self.cash + self.holdings_value(prices)

    def is_invested(self, ticker: str) -> bool:
        return self._positions.get(ticker, 0) > 0

    # ── Trade execution ───────────────────────────────────────────────────────

    def buy(self, ticker: str, shares: int, price: float, date_str: str) -> bool:
        """Execute a buy order. Adjusts share count down if insufficient cash.

        Returns:
            True if at least 1 share was bought, False otherwise.
        """
        if shares <= 0:
            return False

        commission = price * shares * self.commission_pct
        total_cost = price * shares + commission

        if total_cost > self.cash:
            max_shares = int(self.cash / (price * (1.0 + self.commission_pct)))
            if max_shares <= 0:
                return False
            shares = max_shares
            commission = price * shares * self.commission_pct
            total_cost = price * shares + commission

        self.cash -= total_cost

        prev_shares = self._positions.get(ticker, 0)
        prev_cost = self._avg_cost.get(ticker, 0.0)
        new_shares = prev_shares + shares
        self._avg_cost[ticker] = (prev_cost * prev_shares + price * shares) / new_shares
        self._positions[ticker] = new_shares

        self.trades.append(
            Trade(
                date=date_str,
                ticker=ticker,
                action="BUY",
                shares=shares,
                price=price,
                commission=commission,
            )
        )
        return True

    def sell(self, ticker: str, shares: int, price: float, date_str: str) -> bool:
        """Execute a sell order (partial or full).

        Returns:
            True if shares were sold, False otherwise.
        """
        held = self._positions.get(ticker, 0)
        if held <= 0:
            return False

        shares = min(shares, held)
        commission = price * shares * self.commission_pct
        proceeds = price * shares - commission
        pnl = (price - self._avg_cost.get(ticker, price)) * shares - commission

        self.cash += proceeds
        remaining = held - shares
        if remaining <= 0:
            self._positions.pop(ticker, None)
            self._avg_cost.pop(ticker, None)
        else:
            self._positions[ticker] = remaining

        self.trades.append(
            Trade(
                date=date_str,
                ticker=ticker,
                action="SELL",
                shares=shares,
                price=price,
                commission=commission,
                pnl=pnl,
            )
        )
        return True

    def sell_all(self, ticker: str, price: float, date_str: str) -> bool:
        """Liquidate the entire position for a ticker."""
        return self.sell(ticker, self._positions.get(ticker, 0), price, date_str)

    def accrue_cash_interest(self, daily_rf_rate: float) -> None:
        """Add one day of risk-free interest on uninvested cash."""
        if daily_rf_rate > 0 and self.cash > 0:
            self.cash += self.cash * daily_rf_rate

    # ── Equity tracking ───────────────────────────────────────────────────────

    def record_equity(self, date_str: str, prices: Dict[str, float]) -> None:
        """Append a daily equity snapshot."""
        eq = self.equity(prices)
        self.equity_curve.append(
            {
                "date": date_str,
                "equity": eq,
                "cash": self.cash,
                "holdings_value": self.holdings_value(prices),
            }
        )

    def equity_series(self) -> pd.Series:
        """Return equity curve as a pandas Series indexed by date string."""
        if not self.equity_curve:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.equity_curve).set_index("date")
        return df["equity"]

    def to_trades_df(self) -> pd.DataFrame:
        """Return the trade log as a DataFrame."""
        if not self.trades:
            return pd.DataFrame(
                columns=["date", "ticker", "action", "shares", "price", "commission", "pnl"]
            )
        return pd.DataFrame([vars(t) for t in self.trades])
