"""Strategy configuration and parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal


@dataclass
class IndicatorConfig:
    """Moving average (primary) indicator settings."""
    fast_period: int = 20
    slow_period: int = 50
    ma_type: Literal["sma", "ema", "mvwap"] = "mvwap"


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
    use_macd_entry: bool = False  # Option 2: replace MA crossover with EMA-gate + MACD zero-cross
    trend_gate_period: int = 200  # EMA period used as trend gate when use_macd_entry=True
    divergence_filter: bool = True   # Block BUY entry on bearish MACD divergence
    divergence_exit: bool = True     # Force SELL when bearish divergence detected on open position


@dataclass
class ATRStopConfig:
    """ATR-based dynamic stop loss."""
    enabled: bool = False
    period: int = 14
    multiplier: float = 3.5
    trail: bool = True  # Ratchet stop up each day as price rises (never moves down)
    profit_stop_enabled: bool = True  # Trailing profit stop — exit if price falls N×ATR from peak since entry
    profit_stop_atr_mult: float = 3.0  # ATR multiplier for profit stop (wider than stop-loss)
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
    enabled: bool = True
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
class ShortConfig:
    """Short-side trend following — trade bearish trends."""
    enabled: bool = False
    rsi_oversold_required: bool = True  # Confirm SHORT when RSI oversold
    adx_filter: bool = True  # Block SHORT when ADX too weak


@dataclass
class PortfolioConstraintsConfig:
    """Cross-ticker portfolio risk and concentration limits.

    Applied only when using run_portfolio_backtest(). The single-ticker
    run_backtest() uses max_position_pct from PositionSizingConfig instead.
    """
    max_open_positions: int = 10           # max simultaneous positions (long+short); 0 = unlimited
    max_sector_exposure_pct: float = 40.0  # max % of NAV in any one sector; 0 = unlimited
    max_gross_exposure_pct: float = 100.0  # max (long+short notional) / NAV; 0 = unlimited
    adv_participation_pct: float = 2.5     # cap order at this % of daily volume; 0 = unlimited


# Sector classification used by portfolio constraint checks
SECTOR_MAP: Dict[str, str] = {
    # Utilities / Energy — trend plays
    "VST": "Utilities", "GEV": "Industrials", "MP": "Materials", "UUUU": "Materials",
    # Technology — strong trend stocks
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "META": "Technology", "NVDA": "Technology",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    # Financials
    "V": "Financials", "MA": "Financials",
    # Healthcare
    "UNH": "Healthcare", "ABBV": "Healthcare", "MRK": "Healthcare", "JNJ": "Healthcare",
    # Consumer Staples
    "COST": "Consumer Staples",
    # Industrials
    "CAT": "Industrials", "LIN": "Industrials",
    # Technology (mature)
    "IBM": "Technology",
    # Diversified ETFs
    "QQQ": "Technology", "SPY": "Diversified", "SQQQ": "Inverse",
    # Fixed Income
    "TLT": "Fixed Income",
}


@dataclass
class SignalConfig:
    """Signal generation settings."""
    # Sentiment filter thresholds
    sentiment_filter_enabled: bool = True
    min_sentiment_confidence: float = 0.4
    block_on_negative_sentiment: bool = True

    # Risk filter thresholds
    risk_filter_enabled: bool = True
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
    use_kelly_fraction: bool = True
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
    abs_return_hurdle: float = 0.03  # Cash + hurdle benchmark: rf + this rate


@dataclass
class TrendFollowingConfig:
    """Master configuration combining all sub-configs."""
    indicator: IndicatorConfig = field(default_factory=IndicatorConfig)

    # Per-ticker MA overrides — cyclicals use slower EMA 50/200 to avoid whipsaws;
    # tech names (default) use EMA 20/50 for faster entry.
    ticker_indicator_overrides: Dict[str, IndicatorConfig] = field(
        default_factory=lambda: {
            "XOM": IndicatorConfig(fast_period=50, slow_period=200),
            "CAT": IndicatorConfig(fast_period=50, slow_period=200),
            "GLD": IndicatorConfig(fast_period=50, slow_period=200),
            "WMT": IndicatorConfig(fast_period=50, slow_period=200),
            "LLY": IndicatorConfig(fast_period=50, slow_period=200),
            "UNH": IndicatorConfig(fast_period=50, slow_period=200),
            "LIN": IndicatorConfig(fast_period=50, slow_period=200),
            "IBM": IndicatorConfig(fast_period=50, slow_period=200),
        }
    )

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
    short: ShortConfig = field(default_factory=ShortConfig)
    portfolio_constraints: PortfolioConstraintsConfig = field(default_factory=PortfolioConstraintsConfig)
    sector_map: Dict[str, str] = field(default_factory=lambda: dict(SECTOR_MAP))

    # Tickers to trade (mirrors sentiment/risk pipelines)
    tickers: List[str] = field(default_factory=lambda: [
        # Utilities / Energy / Materials — trending sectors
        "VST", "GEV", "MP", "UUUU",
        # Technology — strong trend stocks
        "AAPL", "MSFT", "GOOGL", "META", "NVDA",
        # Consumer Discretionary
        "AMZN", "TSLA", "HD", "MCD",
        # Financials
        "V", "MA",
        # Healthcare
        "UNH", "ABBV", "MRK", "JNJ",
        # Consumer Staples
        "COST",
        # Industrials
        "CAT", "LIN",
        # Technology (mature)
        "IBM",
        # Diversified ETFs
        "QQQ", "SPY",
        # Fixed Income / Inverse — macro trend hedges
        "TLT",
        "SQQQ",
    ])

    lookback_days: int = 7300  # 20 calendar years → ~5040 trading days

    def get_indicator_cfg(self, ticker: str) -> IndicatorConfig:
        """Return per-ticker IndicatorConfig, falling back to the default."""
        return self.ticker_indicator_overrides.get(ticker.upper(), self.indicator)


# Default configuration instance
DEFAULT_CONFIG = TrendFollowingConfig()
