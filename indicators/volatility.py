"""Volatility indicators: ATR (stop loss) and Volatility Regime filter.

ATR (Average True Range):
    - Computes dynamic stop-loss price: entry - N * ATR
    - Used by backtest engine to auto-exit if price breaches stop

Volatility Regime:
    - Measures 30-day realised annualised volatility
    - Reduces position size when vol is high (> threshold)
    - Provides a continuous multiplier [0.25 .. 1.0] for position sizing
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Indicator, IndicatorResult


class ATR(Indicator):
    """Average True Range using Wilder smoothing.

    Returns ATR values as the `values` series.
    signal_series() is not meaningful for ATR — use `stop_price()` instead.
    """

    def __init__(self, period: int = 14) -> None:
        self._period = period

    @property
    def name(self) -> str:
        return f"ATR({self._period})"

    def compute(self, ohlc: pd.DataFrame) -> IndicatorResult:
        high  = ohlc["High"]
        low   = ohlc["Low"]
        close = ohlc["Close"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(alpha=1.0 / self._period, adjust=False).mean()
        atr.iloc[: self._period] = np.nan
        return IndicatorResult(values=atr, name=self.name)

    def signal_series(self, ohlc: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=ohlc.index)

    def latest_atr(self, ohlc: pd.DataFrame) -> float:
        atr = self.compute(ohlc).values.dropna()
        return float(atr.iloc[-1]) if not atr.empty else 0.0

    def stop_price(
        self,
        ohlc: pd.DataFrame,
        entry_price: float,
        multiplier: float = 2.0,
        direction: str = "long",
    ) -> float:
        """Compute ATR-based stop price.

        Args:
            ohlc: OHLCV DataFrame.
            entry_price: Trade entry price.
            multiplier: ATR multiplier (default 2.0).
            direction: 'long' (stop below entry) or 'short' (stop above).

        Returns:
            Stop loss price.
        """
        atr = self.latest_atr(ohlc)
        if atr == 0.0:
            return entry_price * (0.95 if direction == "long" else 1.05)
        if direction == "long":
            return entry_price - multiplier * atr
        return entry_price + multiplier * atr

    def atr_series(self, ohlc: pd.DataFrame) -> pd.Series:
        """Return the full ATR series for use in backtest simulation."""
        return self.compute(ohlc).values


class VolatilityRegime(Indicator):
    """Realised volatility regime filter.

    Computes 30-day annualised realised volatility from log returns.
    Provides a position-size multiplier:
        vol <= low_threshold    → 1.0  (full size)
        vol in (low, high)      → linearly scaled down
        vol >= high_threshold   → min_multiplier (minimal size)

    signal_series() returns the raw multiplier [min_mult .. 1.0].
    """

    def __init__(
        self,
        period: int = 30,
        low_vol_threshold: float = 0.15,
        high_vol_threshold: float = 0.30,
        min_multiplier: float = 0.25,
    ) -> None:
        self._period = period
        self._low = low_vol_threshold
        self._high = high_vol_threshold
        self._min_mult = min_multiplier

    @property
    def name(self) -> str:
        return f"VolRegime({self._period})"

    def compute(self, ohlc: pd.DataFrame) -> IndicatorResult:
        log_ret = np.log(ohlc["Close"] / ohlc["Close"].shift(1))
        vol = log_ret.rolling(self._period, min_periods=self._period).std() * np.sqrt(252)
        return IndicatorResult(values=vol, name=self.name)

    def signal_series(self, ohlc: pd.DataFrame) -> pd.Series:
        """Return size multiplier [min_mult .. 1.0] based on volatility regime."""
        vol = self.compute(ohlc).values

        def _mult(v):
            if pd.isna(v):
                return 1.0
            if v <= self._low:
                return 1.0
            if v >= self._high:
                return self._min_mult
            span = self._high - self._low
            return float(1.0 - (1.0 - self._min_mult) * (v - self._low) / span)

        return vol.map(_mult)

    def latest_multiplier(self, ohlc: pd.DataFrame) -> float:
        """Return the current position-size multiplier based on volatility."""
        sig = self.signal_series(ohlc).dropna()
        return float(sig.iloc[-1]) if not sig.empty else 1.0

    def latest_vol(self, ohlc: pd.DataFrame) -> float:
        vol = self.compute(ohlc).values.dropna()
        return float(vol.iloc[-1]) if not vol.empty else 0.0
