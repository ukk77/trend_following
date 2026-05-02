"""Moving average indicators: SMA, EMA, and CrossoverSignal.

CrossoverSignal is the primary indicator used by the strategy.
SMA and EMA are building blocks that can also be used standalone.

Extensibility: add RSI, MACD, ADX by subclassing Indicator (base.py).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .base import Indicator, IndicatorResult


class SMA(Indicator):
    """Simple Moving Average."""

    def __init__(self, window: int, price_col: str = "Close") -> None:
        self._window = window
        self._price_col = price_col

    @property
    def name(self) -> str:
        return f"SMA({self._window})"

    def compute(self, ohlc: pd.DataFrame) -> IndicatorResult:
        series = ohlc[self._price_col].rolling(self._window, min_periods=self._window).mean()
        return IndicatorResult(values=series, name=self.name)


class EMA(Indicator):
    """Exponential Moving Average."""

    def __init__(self, window: int, price_col: str = "Close") -> None:
        self._window = window
        self._price_col = price_col

    @property
    def name(self) -> str:
        return f"EMA({self._window})"

    def compute(self, ohlc: pd.DataFrame) -> IndicatorResult:
        series = ohlc[self._price_col].ewm(span=self._window, adjust=False).mean()
        return IndicatorResult(values=series, name=self.name)


class CrossoverSignal(Indicator):
    """Trend signal from fast/slow MA crossover.

    Output:
        +1.0  when fast MA > slow MA  (bullish trend)
        -1.0  when fast MA < slow MA  (bearish trend)
         0.0  when equal or insufficient data

    The `raw` DataFrame from `compute()` contains the fast MA, slow MA,
    and gap_pct columns for inspection or plotting.
    """

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 50,
        ma_type: Literal["SMA", "EMA", "sma", "ema"] = "EMA",
        price_col: str = "Close",
    ) -> None:
        if fast_window >= slow_window:
            raise ValueError(
                f"fast_window ({fast_window}) must be < slow_window ({slow_window})"
            )
        self._fast_window = fast_window
        self._slow_window = slow_window
        self._ma_type = ma_type.upper()
        self._price_col = price_col

        MA = EMA if self._ma_type == "EMA" else SMA
        self._fast_ma = MA(fast_window, price_col)
        self._slow_ma = MA(slow_window, price_col)

    @property
    def name(self) -> str:
        return f"Crossover({self._ma_type}{self._fast_window}/{self._slow_window})"

    def compute(self, ohlc: pd.DataFrame) -> IndicatorResult:
        fast = self._fast_ma.compute(ohlc).values
        slow = self._slow_ma.compute(ohlc).values

        with np.errstate(divide="ignore", invalid="ignore"):
            gap_pct = (fast - slow) / slow.replace(0.0, np.nan)

        direction = np.sign(gap_pct).fillna(0.0)

        raw = pd.DataFrame(
            {
                f"fast_{self._ma_type}{self._fast_window}": fast,
                f"slow_{self._ma_type}{self._slow_window}": slow,
                "gap_pct": gap_pct,
                "direction": direction,
            },
            index=ohlc.index,
        )
        return IndicatorResult(values=direction, raw=raw, name=self.name)

    def signal_series(self, ohlc: pd.DataFrame) -> pd.Series:
        return self.compute(ohlc).values

    def get_ma_dataframe(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """Return full DataFrame with MAs and gap for analysis or plotting."""
        return self.compute(ohlc).raw
