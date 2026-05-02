"""Abstract base class for all trend indicators.

New indicators can be added by subclassing Indicator and implementing
`compute()`. The `signal_series()` method provides a normalised
[-1, +1] output usable by the signal generator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class IndicatorResult:
    """Standardised output container for any indicator."""
    values: pd.Series
    raw: Optional[pd.DataFrame] = None
    name: str = ""


class Indicator(ABC):
    """Common interface for all trend indicators.

    Subclasses must implement `compute(ohlc)`.
    `signal_series()` returns a float series in [-1 (bearish), +1 (bullish)],
    which can be overridden for richer semantics.
    """

    @abstractmethod
    def compute(self, ohlc: pd.DataFrame) -> IndicatorResult:
        """Compute the indicator from OHLCV DataFrame.

        Args:
            ohlc: DataFrame with columns Open, High, Low, Close, Volume.
                  Index must be datetime.

        Returns:
            IndicatorResult with at minimum a `.values` Series.
        """
        ...

    def signal_series(self, ohlc: pd.DataFrame) -> pd.Series:
        """Return a normalised signal series in [-1.0, +1.0].

        Default implementation: sign of `values`.
        Override in subclasses for richer signal calculation.
        """
        result = self.compute(ohlc)
        return result.values.map(
            lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable indicator name."""
        ...
