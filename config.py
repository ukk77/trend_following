"""Strategy configuration and parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class IndicatorConfig:
    """Moving average (primary) indicator settings."""
    fast_period: int = 50
    slow_period: int = 200
    ma_type: Literal["sma", "ema"] = "ema"


@dataclass
class RSIConfig:
    """RSI overbought/oversold filter."""
    enabled: bool = True
    period: int = 14
    overbought: float = 70.0
    oversold: float = 30.0


@dataclass
class ADXConfig:
    """ADX trend-strength filter — only trade when trend is confirmed."""
    enabled: bool = True
    period: int = 14
    min_adx: float = 25.0


@dataclass
class MACDConfig:
    """MACD histogram confirmation — secondary momentum check."""
    enabled: bool = True
    fast: int = 12
    slow: int = 26
    signal: int = 9


@dataclass
class ATRStopConfig:
    """ATR-based dynamic stop loss."""
    enabled: bool = True
    period: int = 14
    multiplier: float = 2.0
    trail: bool = True  # Ratchet stop up each day as price rises (never moves down)
    use_db_stop_when_available: bool = False  # Prefer suggested_stop_loss_pct from risk DB; fall back to local ATR


@dataclass
class VolumeConfig:
    """Volume confirmation — require above-average volume on signal."""
    enabled: bool = True
    period: int = 20
    min_ratio: float = 1.0


@dataclass
class RangeFilterConfig:
    """52-week range filter — block BUY when price is overextended."""
    enabled: bool = False
    lookback_days: int = 252
    top_block_pct: float = 0.90


@dataclass
class VolatilityRegimeConfig:
    """Volatility regime — scale down position size in high-vol periods."""
    enabled: bool = False
    period: int = 30
    low_vol_threshold: float = 0.15
    high_vol_threshold: float = 0.30
    min_multiplier: float = 0.25


@dataclass
class MultiTimeframeConfig:
    """Weekly trend confirmation — only take signals when weekly agrees."""
    enabled: bool = True
    fast_weeks: int = 4
    slow_weeks: int = 10


@dataclass
class SignalConfig:
    """Signal generation settings."""
    # Sentiment filter thresholds
    sentiment_filter_enabled: bool = False
    min_sentiment_confidence: float = 0.4
    block_on_negative_sentiment: bool = True

    # Risk filter thresholds
    risk_filter_enabled: bool = False
    max_risk_score: float = 75.0  # Skip BUY if composite_risk_score > this


@dataclass
class PositionSizingConfig:
    """Position sizing settings."""
    base_position_pct: float = 10.0  # Base position as % of portfolio
    max_position_pct: float = 20.0   # Max single position
    
    # Sentiment alignment multipliers
    sentiment_agree_mult: float = 1.2    # Trend + sentiment agree
    sentiment_neutral_mult: float = 0.8  # Trend + sentiment neutral
    sentiment_disagree_mult: float = 0.5 # Trend + sentiment disagree (reduce, not skip)
    
    # Use Kelly fraction from risk calculator if available
    use_kelly_fraction: bool = False
    kelly_cap: float = 0.25  # Max Kelly fraction to use

    # Scale position by vol_ann_30d/vol_ann_1y ratio from risk DB (disabled until DB has sufficient history)
    vol_regime_db_enabled: bool = False


@dataclass
class BacktestConfig:
    """Backtest engine settings."""
    initial_capital: float = 100_000.0
    commission_per_trade: float = 0.0  # Flat fee per trade
    commission_pct: float = 0.001      # 0.1% per trade (slippage + commission)
    slippage: float = 0.0005           # One-way slippage applied on execution
    
    model_cash_interest: bool = True  # Accrue T-bill yield on uninvested cash

    # Benchmark tickers
    benchmark_ticker: str = "SPY"
    compare_buy_and_hold: bool = True  # Also compare vs B&H of each ticker


@dataclass
class TrendFollowingConfig:
    """Master configuration combining all sub-configs."""
    indicator: IndicatorConfig = field(default_factory=IndicatorConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Additional indicators
    rsi: RSIConfig = field(default_factory=RSIConfig)
    adx: ADXConfig = field(default_factory=ADXConfig)
    macd: MACDConfig = field(default_factory=MACDConfig)
    atr_stop: ATRStopConfig = field(default_factory=ATRStopConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    range_filter: RangeFilterConfig = field(default_factory=RangeFilterConfig)
    vol_regime: VolatilityRegimeConfig = field(default_factory=VolatilityRegimeConfig)
    mtf: MultiTimeframeConfig = field(default_factory=MultiTimeframeConfig)

    # Tickers to trade (mirrors sentiment/risk pipelines)
    tickers: List[str] = field(default_factory=lambda: [
        # Tech / Communication
        "AAPL", "MSFT", "GOOGL", "META", "NVDA",
        # Consumer Discretionary
        "AMZN", "TSLA",
        # Financials
        "JPM",
        # Energy
        "XOM",
        # Healthcare
        "LLY", "UNH",
        # Consumer Staples
        "WMT",
        # Industrials
        "CAT",
    ])

    lookback_days: int = 7300  # 20 calendar years → ~5040 trading days


# Default configuration instance
DEFAULT_CONFIG = TrendFollowingConfig()
