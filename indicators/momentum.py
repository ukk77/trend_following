"""Momentum indicators: RSI and MACD.

RSI  — Relative Strength Index (Wilder smoothing)
MACD — Moving Average Convergence Divergence

Both subclass Indicator and can be used standalone or as filters in the
signal generator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Indicator, IndicatorResult


class RSI(Indicator):
    """Relative Strength Index using Wilder's smoothing (EWMA with alpha=1/period).

    signal_series() returns:
        +1.0  when RSI <= oversold  (oversold — potential BUY)
        -1.0  when RSI >= overbought (overbought — potential SELL / block BUY)
         0.0  neutral zone
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        price_col: str = "Close",
    ) -> None:
        self._period = period
        self._overbought = overbought
        self._oversold = oversold
        self._price_col = price_col

    @property
    def name(self) -> str:
        return f"RSI({self._period})"

    def compute(self, ohlc: pd.DataFrame) -> IndicatorResult:
        delta = ohlc[self._price_col].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        alpha = 1.0 / self._period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi.iloc[: self._period] = np.nan

        return IndicatorResult(values=rsi, name=self.name)

    def signal_series(self, ohlc: pd.DataFrame) -> pd.Series:
        rsi = self.compute(ohlc).values
        sig = pd.Series(0.0, index=rsi.index)
        sig[rsi >= self._overbought] = -1.0
        sig[rsi <= self._oversold] = 1.0
        return sig

    def is_overbought(self, ohlc: pd.DataFrame) -> bool:
        """Return True if the latest RSI is overbought."""
        rsi = self.compute(ohlc).values.dropna()
        return bool(not rsi.empty and float(rsi.iloc[-1]) >= self._overbought)

    def is_oversold(self, ohlc: pd.DataFrame) -> bool:
        """Return True if the latest RSI is oversold."""
        rsi = self.compute(ohlc).values.dropna()
        return bool(not rsi.empty and float(rsi.iloc[-1]) <= self._oversold)

    def latest_value(self, ohlc: pd.DataFrame) -> float:
        rsi = self.compute(ohlc).values.dropna()
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0


class MACD(Indicator):
    """Moving Average Convergence Divergence.

    Components:
        macd_line    = EMA(fast) - EMA(slow)
        signal_line  = EMA(macd_line, signal_period)
        histogram    = macd_line - signal_line

    signal_series() returns:
        +1.0  when histogram > 0 (bullish momentum)
        -1.0  when histogram < 0 (bearish momentum)
         0.0  at crossover or insufficient data
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        price_col: str = "Close",
    ) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self._fast = fast
        self._slow = slow
        self._signal = signal
        self._price_col = price_col

    @property
    def name(self) -> str:
        return f"MACD({self._fast},{self._slow},{self._signal})"

    def compute(self, ohlc: pd.DataFrame) -> IndicatorResult:
        price = ohlc[self._price_col]
        ema_fast = price.ewm(span=self._fast, adjust=False).mean()
        ema_slow = price.ewm(span=self._slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self._signal, adjust=False).mean()
        histogram = macd_line - signal_line

        raw = pd.DataFrame(
            {
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": histogram,
            },
            index=ohlc.index,
        )
        return IndicatorResult(values=histogram, raw=raw, name=self.name)

    def signal_series(self, ohlc: pd.DataFrame) -> pd.Series:
        hist = self.compute(ohlc).values
        return hist.map(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))

    def histogram_confirms_buy(self, ohlc: pd.DataFrame) -> bool:
        """Return True if MACD histogram is positive (bullish)."""
        hist = self.compute(ohlc).values.dropna()
        return bool(not hist.empty and float(hist.iloc[-1]) > 0)

    def histogram_confirms_sell(self, ohlc: pd.DataFrame) -> bool:
        """Return True if MACD histogram is negative (bearish)."""
        hist = self.compute(ohlc).values.dropna()
        return bool(not hist.empty and float(hist.iloc[-1]) < 0)

    def check_bearish_divergence(self, ohlc: pd.DataFrame, lookback: int = 20) -> bool:
        """
        Detect Bearish MACD Divergence.
        Price makes a Higher High, but MACD histogram makes a Lower High.
        Used as an early trend exhaustion warning (Exit signal).
        """
        hist = self.compute(ohlc).values.dropna()
        if len(hist) < lookback + 5:
            return False
            
        prices = ohlc[self._price_col].loc[hist.index]
        
        # Look at last 'lookback' bars
        recent_prices = prices.iloc[-lookback:]
        recent_hist = hist.iloc[-lookback:]
        
        # If current price is near the high of the lookback period
        if prices.iloc[-1] >= recent_prices.max() * 0.995:
            # Find the absolute max of MACD histogram in the lookback period
            max_hist = recent_hist.max()
            
            # If current MACD histogram is significantly lower than the max MACD peak
            # and current histogram is sloping down (current < prev)
            if hist.iloc[-1] < hist.iloc[-2] and hist.iloc[-1] < (max_hist * 0.75):
                return True
                
        return False

    def bearish_divergence_series(self, ohlc: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Pre-compute a boolean Series: True on bars with bearish MACD divergence."""
        hist = self.compute(ohlc).values
        price = ohlc[self._price_col]
        div = pd.Series(False, index=ohlc.index)
        for i in range(lookback + 5, len(ohlc)):
            h_slice = hist.iloc[i - lookback: i + 1].dropna()
            if len(h_slice) < 3:
                continue
            h_cur, h_prev = hist.iloc[i], hist.iloc[i - 1]
            if pd.isna(h_cur) or pd.isna(h_prev):
                continue
            p_slice = price.iloc[i - lookback: i + 1]
            if float(price.iloc[i]) >= float(p_slice.max()) * 0.995:
                if float(h_cur) < float(h_prev) and float(h_cur) < float(h_slice.max()) * 0.75:
                    div.iloc[i] = True
        return div
