"""Trend strength indicator: ADX (Average Directional Index).

ADX measures trend strength regardless of direction:
    ADX < 20   — no clear trend (sideways / avoid signals)
    ADX 20-25  — weak trend forming
    ADX > 25   — trending market (take signals)
    ADX > 50   — very strong trend

Only the ADX value is used for filtering; +DI/-DI are available
via `compute().raw` for inspection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Indicator, IndicatorResult


class ADX(Indicator):
    """Average Directional Index using Wilder smoothing.

    signal_series() is NOT directional — it returns:
        +1.0  when ADX >= threshold  (trend is strong enough to trade)
         0.0  when ADX <  threshold  (no trend — filter out signals)
    """

    def __init__(self, period: int = 14, threshold: float = 25.0) -> None:
        self._period = period
        self._threshold = threshold

    @property
    def name(self) -> str:
        return f"ADX({self._period})"

    def compute(self, ohlc: pd.DataFrame) -> IndicatorResult:
        high = ohlc["High"]
        low  = ohlc["Low"]
        close = ohlc["Close"]

        # True Range
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        # Directional movement
        up_move   = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        alpha = 1.0 / self._period

        atr      = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di  = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)

        di_sum  = plus_di + minus_di
        dx = (100.0 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)).fillna(0.0)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        # Mask first period rows
        adx.iloc[: self._period * 2] = np.nan

        raw = pd.DataFrame(
            {
                "adx": adx,
                "+di": plus_di,
                "-di": minus_di,
                "atr": atr,
            },
            index=ohlc.index,
        )
        return IndicatorResult(values=adx, raw=raw, name=self.name)

    def signal_series(self, ohlc: pd.DataFrame) -> pd.Series:
        adx = self.compute(ohlc).values
        return adx.map(lambda x: 1.0 if (x is not None and not np.isnan(x) and x >= self._threshold) else 0.0)

    def is_trending(self, ohlc: pd.DataFrame) -> bool:
        """Return True if the latest ADX >= threshold (trend strong enough)."""
        adx = self.compute(ohlc).values.dropna()
        return bool(not adx.empty and float(adx.iloc[-1]) >= self._threshold)

    def latest_value(self, ohlc: pd.DataFrame) -> float:
        adx = self.compute(ohlc).values.dropna()
        return float(adx.iloc[-1]) if not adx.empty else 0.0
