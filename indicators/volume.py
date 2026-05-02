"""Volume confirmation indicator.

Confirms trend signals only when volume is above average — filters out
low-conviction moves that lack participation.

Logic:
    current_volume > volume_ma(period) * min_ratio  →  confirmed
    otherwise                                        →  unconfirmed (block signal)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Indicator, IndicatorResult


class VolumeConfirmation(Indicator):
    """Volume-based signal confirmation filter.

    signal_series() returns:
        +1.0  when latest volume > MA(volume) * min_ratio  (confirmed)
         0.0  otherwise                                      (unconfirmed)
    """

    def __init__(
        self,
        period: int = 20,
        min_ratio: float = 1.0,
    ) -> None:
        self._period = period
        self._min_ratio = min_ratio

    @property
    def name(self) -> str:
        return f"VolumeConfirm({self._period})"

    def compute(self, ohlc: pd.DataFrame) -> IndicatorResult:
        vol = ohlc["Volume"].astype(float)
        vol_ma = vol.rolling(self._period, min_periods=self._period).mean()

        raw = pd.DataFrame(
            {"volume": vol, "volume_ma": vol_ma, "ratio": vol / vol_ma.replace(0, np.nan)},
            index=ohlc.index,
        )
        return IndicatorResult(values=vol_ma, raw=raw, name=self.name)

    def signal_series(self, ohlc: pd.DataFrame) -> pd.Series:
        result = self.compute(ohlc)
        ratio = result.raw["ratio"] if result.raw is not None else pd.Series(dtype=float)
        return ratio.map(
            lambda x: 1.0 if (x is not None and not np.isnan(x) and x >= self._min_ratio) else 0.0
        )

    def is_confirmed(self, ohlc: pd.DataFrame) -> bool:
        """Return True if latest volume confirms the signal."""
        sig = self.signal_series(ohlc).dropna()
        return bool(not sig.empty and float(sig.iloc[-1]) >= 1.0)

    def latest_ratio(self, ohlc: pd.DataFrame) -> float:
        """Return current volume / MA(volume) ratio."""
        result = self.compute(ohlc)
        if result.raw is None:
            return 0.0
        ratio = result.raw["ratio"].dropna()
        return float(ratio.iloc[-1]) if not ratio.empty else 0.0
