"""Range and multi-timeframe filters.

RangeFilter:
    Blocks BUY when price is in the top X% of its 52-week range
    (overextended — buying at peak, poor risk/reward).

MultiTimeframeConfirmation:
    Resamples daily OHLCV to weekly bars and runs a slow EMA crossover.
    Only takes daily signals when the weekly trend agrees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Indicator, IndicatorResult


class RangeFilter(Indicator):
    """52-week range position filter.

    Computes: position = (close - low_52w) / (high_52w - low_52w)
        0.0 = at 52-week low
        1.0 = at 52-week high

    signal_series() returns:
        +1.0  when position <= top_block_threshold  (not overextended — allow BUY)
         0.0  when position >  top_block_threshold  (overextended — block BUY)
    """

    def __init__(
        self,
        lookback_days: int = 252,
        top_block_threshold: float = 0.90,
    ) -> None:
        self._lookback = lookback_days
        self._top_block = top_block_threshold

    @property
    def name(self) -> str:
        return f"RangeFilter({self._lookback}d,top>{self._top_block:.0%})"

    def compute(self, ohlc: pd.DataFrame) -> IndicatorResult:
        close = ohlc["Close"]
        low_52w  = close.rolling(self._lookback, min_periods=self._lookback // 2).min()
        high_52w = close.rolling(self._lookback, min_periods=self._lookback // 2).max()

        span = high_52w - low_52w
        position = (close - low_52w) / span.replace(0, np.nan)
        return IndicatorResult(values=position, name=self.name)

    def signal_series(self, ohlc: pd.DataFrame) -> pd.Series:
        pos = self.compute(ohlc).values
        return pos.map(
            lambda x: 1.0 if (x is not None and not np.isnan(x) and x <= self._top_block) else 0.0
        )

    def is_overextended(self, ohlc: pd.DataFrame) -> bool:
        """Return True if price is in the top bucket of its 52-week range."""
        pos = self.compute(ohlc).values.dropna()
        return bool(not pos.empty and float(pos.iloc[-1]) > self._top_block)

    def latest_position(self, ohlc: pd.DataFrame) -> float:
        """Return 0..1 range position (1 = at 52-week high)."""
        pos = self.compute(ohlc).values.dropna()
        return float(pos.iloc[-1]) if not pos.empty else 0.5


class MultiTimeframeConfirmation(Indicator):
    """Weekly trend confirmation filter.

    Resamples daily OHLCV to weekly bars, computes EMA crossover on the
    weekly timeframe. Only allows signals when weekly trend agrees with
    the daily trend direction.

    signal_series() returns:
        +1.0  weekly bullish (fast EMA > slow EMA on weekly)
        -1.0  weekly bearish
         0.0  insufficient weekly data
    """

    def __init__(
        self,
        fast_weeks: int = 4,
        slow_weeks: int = 10,
    ) -> None:
        self._fast = fast_weeks
        self._slow = slow_weeks

    @property
    def name(self) -> str:
        return f"MTF(W{self._fast}/W{self._slow})"

    def _resample_weekly(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily OHLCV to weekly (Monday-aligned, last bar of week)."""
        agg = {
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }
        available = {k: v for k, v in agg.items() if k in ohlc.columns}
        weekly = ohlc.resample("W").agg(available).dropna(subset=["Close"])
        return weekly

    def compute(self, ohlc: pd.DataFrame) -> IndicatorResult:
        weekly = self._resample_weekly(ohlc)
        if len(weekly) < self._slow:
            empty = pd.Series(np.nan, index=ohlc.index)
            return IndicatorResult(values=empty, name=self.name)

        fast_ema = weekly["Close"].ewm(span=self._fast, adjust=False).mean()
        slow_ema = weekly["Close"].ewm(span=self._slow, adjust=False).mean()
        weekly_dir = np.sign(fast_ema - slow_ema)

        # Reindex weekly signal back to daily (forward-fill)
        daily_dir = weekly_dir.reindex(ohlc.index, method="ffill")
        return IndicatorResult(values=daily_dir, name=self.name)

    def signal_series(self, ohlc: pd.DataFrame) -> pd.Series:
        return self.compute(ohlc).values.fillna(0.0)

    def weekly_trend(self, ohlc: pd.DataFrame) -> float:
        """Return +1 (bull), -1 (bear), or 0 (unknown) for the current weekly trend."""
        sig = self.signal_series(ohlc).dropna()
        if sig.empty:
            return 0.0
        return float(sig.iloc[-1])
