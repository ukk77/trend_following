"""Core backtest engine for the Trend Following strategy.

Each ticker is simulated in its own isolated portfolio (independent capital).
A combined portfolio equity curve is produced by equal-weight averaging the
normalised returns across all tickers.

Sentiment + risk data are loaded from the existing SQLite history DBs and
aligned to each trading date using point-in-time lookups (no look-ahead bias).

Usage:
    from trend_following.backtest.engine import run_backtest, BacktestSummary
    from trend_following.config import TrendFollowingConfig

    summary = run_backtest(cfg, ticker_ohlc, benchmark_ohlc)
    for ticker, result in summary.results.items():
        print(ticker, result.metrics)
"""
from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_TRADING_ROOT = Path(__file__).resolve().parents[2]
_RISK_BACKEND = _TRADING_ROOT / "risk_calculator" / "backend"
if str(_RISK_BACKEND) not in sys.path:
    sys.path.insert(0, str(_RISK_BACKEND))

from ..config import TrendFollowingConfig
from ..indicators.moving_averages import CrossoverSignal, EMA as EMAIndicator
from ..indicators.momentum import RSI, MACD
from ..indicators.trend_strength import ADX
from ..indicators.volatility import ATR, VolatilityRegime
from ..indicators.volume import VolumeConfirmation
from ..indicators.range_filter import RangeFilter, MultiTimeframeConfirmation
from ..position_sizing.sizer import shares_to_buy
from ..signals.generator import Action, Signal
from ..signals.filters import apply_filters
from .metrics import compute_all_metrics
from .portfolio import Portfolio


# ── History DB helpers ────────────────────────────────────────────────────────

def _load_sentiment_history(ticker: str) -> pd.DataFrame:
    """Load all sentiment snapshots for a ticker as a date-indexed DataFrame."""
    db_path = _TRADING_ROOT / "sentiment_analysis" / "backend" / "sentiment_history.db"
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(
                "SELECT captured_at, overall_sentiment, confidence "
                "FROM sentiment_snapshots WHERE UPPER(ticker)=UPPER(?)",
                conn,
                params=(ticker.upper(),),
            )
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["captured_at"]).dt.date
        df = df.sort_values("date").drop_duplicates("date", keep="last")
        return df.set_index("date")
    except Exception:
        return pd.DataFrame()


def _load_risk_history(ticker: str) -> pd.DataFrame:
    """Load all risk snapshots for a ticker as a date-indexed DataFrame."""
    db_path = _TRADING_ROOT / "risk_calculator" / "backend" / "risk_history.db"
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(
                "SELECT captured_at, composite_risk_score, risk_bucket, "
                "kelly_fraction_capped, suggested_stop_loss_pct "
                "FROM risk_snapshots WHERE UPPER(ticker)=UPPER(?)",
                conn,
                params=(ticker.upper(),),
            )
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["captured_at"]).dt.date
        df = df.sort_values("date").drop_duplicates("date", keep="last")
        return df.set_index("date")
    except Exception:
        return pd.DataFrame()


def _get_as_of(df: pd.DataFrame, as_of_date) -> Optional[dict]:
    """Return the most recent row up to and including as_of_date."""
    if df.empty:
        return None
    past = df[df.index <= as_of_date]
    if past.empty:
        return None
    return past.iloc[-1].to_dict()


def _cash_hurdle_equity(
    equity: pd.Series,
    initial_capital: float,
    rf_annual: float,
    hurdle: float = 0.03,
) -> pd.Series:
    """Synthetic benchmark growing at (rf_annual + hurdle) per year."""
    n = len(equity)
    if n == 0:
        return pd.Series(dtype=float)
    daily_factor = (1.0 + rf_annual + hurdle) ** (1.0 / 252)
    vals = initial_capital * (daily_factor ** np.arange(1, n + 1))
    return pd.Series(vals.astype(float), index=equity.index)


def _val_at(series: Optional[pd.Series], dt, default=None):
    """Safe point-in-time value lookup (module-level helper for portfolio backtest)."""
    if series is None or dt not in series.index:
        return default
    v = series.loc[dt]
    return default if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Per-ticker backtest output."""
    ticker: str
    equity_curve: pd.Series
    trades_df: pd.DataFrame
    metrics: Dict
    benchmark_equity: Dict[str, pd.Series]


@dataclass
class BacktestSummary:
    """Aggregated backtest output across all tickers."""
    results: Dict[str, BacktestResult] = field(default_factory=dict)
    portfolio_equity: Optional[pd.Series] = None
    portfolio_metrics: Optional[Dict] = None


# ── Main engine ───────────────────────────────────────────────────────────────

def _run_single_ticker(
    ticker: str,
    ohlc: pd.DataFrame,
    cfg: TrendFollowingConfig,
    sentiment_hist: pd.DataFrame,
    risk_hist: pd.DataFrame,
    benchmark_ohlc: Dict[str, pd.DataFrame],
    rf_annual: float,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Optional[BacktestResult]:
    """Run the backtest for a single ticker. Returns None if insufficient data."""

    # Filter to requested date range
    if start_date:
        ohlc = ohlc[ohlc.index >= pd.Timestamp(start_date)]
    if end_date:
        ohlc = ohlc[ohlc.index <= pd.Timestamp(end_date)]

    ind_cfg = cfg.get_indicator_cfg(ticker)
    min_bars = max(
        ind_cfg.slow_period, cfg.adx.period * 2,
        cfg.rsi.period, cfg.macd.slow + cfg.macd.signal,
        cfg.volume.period, cfg.atr_stop.period,
        cfg.range_filter.lookback_days if cfg.range_filter.enabled else 0,
    ) + 10
    if ohlc.empty or len(ohlc) < min_bars:
        return None

    # ── Precompute all indicator series (no look-ahead bias) ─────────────────
    if cfg.macd.use_macd_entry:
        ema_gate = EMAIndicator(cfg.macd.trend_gate_period).compute(ohlc).values
        macd_prim = MACD(
            fast=cfg.macd.fast, slow=cfg.macd.slow, signal=cfg.macd.signal
        ).compute(ohlc).values
        close_s = ohlc["Close"]
        valid = (~ema_gate.isna()) & (~macd_prim.isna())
        above = close_s > ema_gate
        full_direction = pd.Series(np.nan, index=ohlc.index)
        full_direction[valid & above & (macd_prim > 0)] = 1.0
        full_direction[valid & (~above | (macd_prim <= 0))] = -1.0
    else:
        full_direction = CrossoverSignal(
            fast_window=ind_cfg.fast_period,
            slow_window=ind_cfg.slow_period,
            ma_type=ind_cfg.ma_type,
        ).signal_series(ohlc)

    adx_series = (
        ADX(period=cfg.adx.period, threshold=cfg.adx.min_adx).compute(ohlc).values
        if cfg.adx.enabled else None
    )
    rsi_series = (
        RSI(period=cfg.rsi.period, overbought=cfg.rsi.overbought).compute(ohlc).values
        if cfg.rsi.enabled else None
    )
    macd_hist_series = (
        MACD(fast=cfg.macd.fast, slow=cfg.macd.slow, signal=cfg.macd.signal).compute(ohlc).values
        if cfg.macd.enabled else None
    )
    vol_ratio_series = (
        VolumeConfirmation(period=cfg.volume.period, min_ratio=cfg.volume.min_ratio)
        .compute(ohlc).raw["ratio"]
        if cfg.volume.enabled else None
    )
    range_pos_series = (
        RangeFilter(
            lookback_days=cfg.range_filter.lookback_days,
            top_block_threshold=cfg.range_filter.top_block_pct,
        ).compute(ohlc).values
        if cfg.range_filter.enabled else None
    )
    mtf_series = (
        MultiTimeframeConfirmation(
            fast_weeks=cfg.mtf.fast_weeks, slow_weeks=cfg.mtf.slow_weeks
        ).signal_series(ohlc)
        if cfg.mtf.enabled else None
    )
    vol_regime_series = (
        VolatilityRegime(
            period=cfg.vol_regime.period,
            low_vol_threshold=cfg.vol_regime.low_vol_threshold,
            high_vol_threshold=cfg.vol_regime.high_vol_threshold,
            min_multiplier=cfg.vol_regime.min_multiplier,
        ).signal_series(ohlc)
        if cfg.vol_regime.enabled else None
    )
    atr_series = (
        ATR(period=cfg.atr_stop.period).atr_series(ohlc)
        if cfg.atr_stop.enabled else None
    )

    def _val(series, dt, default=None):
        """Safe point-in-time value lookup."""
        if series is None or dt not in series.index:
            return default
        v = series.loc[dt]
        return default if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    portfolio = Portfolio(cfg.backtest.initial_capital, cfg.backtest.commission_pct)
    slippage = cfg.backtest.slippage
    daily_rf = (1.0 + rf_annual) ** (1.0 / 252) - 1.0

    # Track per-position ATR stop prices
    atr_stops: dict = {}
    peak_prices: dict = {}  # ticker -> highest price since entry (for profit stop)
    short_stops: dict = {}  # ticker -> short stop (stop-out if price >= stop)

    valid_dates = full_direction.dropna().index

    for dt in valid_dates:
        dt_date = dt.date()
        date_str = dt_date.isoformat()
        current_price = float(ohlc.loc[dt, "Close"])
        last_direction = float(full_direction.loc[dt])
        daily_volume = float(ohlc.loc[dt, "Volume"]) if "Volume" in ohlc.columns else None

        # ── Cash interest on idle capital ──────────────────────────────────
        if cfg.backtest.model_cash_interest:
            portfolio.accrue_cash_interest(daily_rf)

        # Point-in-time sentiment + risk
        sent_snap = _get_as_of(sentiment_hist, dt_date)
        risk_snap = _get_as_of(risk_hist, dt_date)
        overall_sentiment = (sent_snap or {}).get("overall_sentiment")
        conf = float((sent_snap or {}).get("confidence") or 0.0)
        risk_score = (risk_snap or {}).get("composite_risk_score")
        kelly_fraction = (risk_snap or {}).get("kelly_fraction_capped")
        db_suggested_stop_pct = (risk_snap or {}).get("suggested_stop_loss_pct")

        # ── ATR stop check and trail update on open positions ────────────────
        if cfg.atr_stop.enabled and portfolio.is_invested(ticker):
            if ticker in atr_stops and current_price <= atr_stops[ticker]:
                exec_price = current_price * (1.0 - slippage)
                portfolio.sell_all(ticker, exec_price, date_str)
                atr_stops.pop(ticker, None)
                peak_prices.pop(ticker, None)
                portfolio.record_equity(date_str, {ticker: current_price})
                continue

            # Update peak price
            if ticker in peak_prices:
                peak_prices[ticker] = max(peak_prices[ticker], current_price)

            # Trailing profit stop — exit if price falls N×ATR from peak
            if cfg.atr_stop.profit_stop_enabled and ticker in peak_prices:
                atr_val = _val(atr_series, dt, 0.0)
                if atr_val > 0:
                    profit_stop = peak_prices[ticker] - cfg.atr_stop.profit_stop_atr_mult * atr_val
                    if current_price <= profit_stop:
                        exec_price = current_price * (1.0 - slippage)
                        portfolio.sell_all(ticker, exec_price, date_str)
                        atr_stops.pop(ticker, None)
                        peak_prices.pop(ticker, None)
                        portfolio.record_equity(date_str, {ticker: current_price})
                        continue
            # Trail: ratchet stop up as price rises — never moves down
            if cfg.atr_stop.trail:
                atr_val = _val(atr_series, dt, 0.0)
                if atr_val > 0:
                    candidate = current_price - cfg.atr_stop.multiplier * atr_val
                    if ticker in atr_stops:
                        atr_stops[ticker] = max(atr_stops[ticker], candidate)

        # ── ATR stop check on short positions ────────────────────────────────
        if cfg.atr_stop.enabled and portfolio.is_short(ticker):
            if ticker in short_stops and current_price >= short_stops[ticker]:
                exec_price = current_price * (1.0 + slippage)
                portfolio.cover_all(ticker, exec_price, date_str)
                short_stops.pop(ticker, None)
                portfolio.record_equity(date_str, {ticker: current_price})
                continue

        # ── Build filtered action ─────────────────────────────────────────────
        if last_direction > 0:
            filtered_action: Action = "BUY"
        elif last_direction < 0:
            filtered_action = "SHORT" if cfg.short.enabled else "SELL"
        else:
            filtered_action = "HOLD"

        # ── Shared filter pipeline ──────────────────────────────────────
        filtered_action, _filter_reasons = apply_filters(
            raw_action=filtered_action,
            last_direction=last_direction,
            cfg=cfg,
            adx_val=_val(adx_series, dt) if cfg.adx.enabled else None,
            rsi_val=_val(rsi_series, dt) if cfg.rsi.enabled else None,
            macd_hist=_val(macd_hist_series, dt) if cfg.macd.enabled else None,
            vol_ratio=_val(vol_ratio_series, dt) if cfg.volume.enabled else None,
            range_pos=_val(range_pos_series, dt) if cfg.range_filter.enabled else None,
            weekly_trend=_val(mtf_series, dt) if cfg.mtf.enabled else None,
            sentiment_data=sent_snap,
            risk_data=risk_snap,
        )

        # ── Vol regime multiplier ─────────────────────────────────────────────
        vol_mult = _val(vol_regime_series, dt, 1.0) if cfg.vol_regime.enabled else 1.0

        # ── Execution ─────────────────────────────────────────────────────────
        if filtered_action == "BUY":
            exec_price = current_price * (1.0 + slippage)
        elif filtered_action in ("SELL", "SHORT"):
            exec_price = current_price * (1.0 - slippage)
        elif filtered_action == "COVER":
            exec_price = current_price * (1.0 + slippage)
        else:
            exec_price = current_price

        sig = Signal(
            ticker=ticker,
            date=date_str,
            action=filtered_action,
            trend_direction=last_direction,
            filtered_strength=abs(last_direction) * vol_mult,
            reason="",
            sentiment=overall_sentiment,
            sentiment_confidence=conf if conf > 0 else None,
            risk_score=risk_score,
        )

        current_portfolio_value = portfolio.equity({ticker: current_price})

        if filtered_action == "BUY" and not portfolio.is_invested(ticker):
            n_shares = shares_to_buy(sig, current_portfolio_value, exec_price, cfg,
                                     kelly_fraction=kelly_fraction, daily_volume=daily_volume)
            if n_shares > 0 and portfolio.buy(ticker, n_shares, exec_price, date_str):
                peak_prices[ticker] = exec_price
                # Set stop: prefer DB suggested_stop_loss_pct, fall back to local ATR
                if cfg.atr_stop.enabled:
                    stop_price = None
                    if cfg.atr_stop.use_db_stop_when_available and db_suggested_stop_pct is not None:
                        stop_price = exec_price * (1.0 + db_suggested_stop_pct)
                    if stop_price is None:
                        atr_val = _val(atr_series, dt, 0.0)
                        if atr_val > 0:
                            stop_price = exec_price - cfg.atr_stop.multiplier * atr_val
                    if stop_price is not None:
                        atr_stops[ticker] = stop_price

        elif filtered_action == "SELL" and portfolio.is_invested(ticker):
            portfolio.sell_all(ticker, exec_price, date_str)
            atr_stops.pop(ticker, None)

        elif filtered_action == "SHORT" and not portfolio.is_short(ticker) and not portfolio.is_invested(ticker):
            n_shares = shares_to_buy(sig, current_portfolio_value, exec_price, cfg,
                                     kelly_fraction=kelly_fraction, daily_volume=daily_volume)
            if n_shares > 0 and portfolio.short(ticker, n_shares, exec_price, date_str):
                if cfg.atr_stop.enabled:
                    atr_val = _val(atr_series, dt, 0.0)
                    if atr_val > 0:
                        short_stops[ticker] = exec_price + cfg.atr_stop.multiplier * atr_val

        elif filtered_action == "COVER" and portfolio.is_short(ticker):
            portfolio.cover_all(ticker, exec_price, date_str)
            short_stops.pop(ticker, None)

        portfolio.record_equity(date_str, {ticker: current_price})

    trades_df = portfolio.to_trades_df()
    equity = portfolio.equity_series()

    # Build benchmark equity curves normalised to initial capital
    bench_equities: Dict[str, pd.Series] = {}
    for b_name, b_ohlc in benchmark_ohlc.items():
        b_filtered = b_ohlc.copy()
        if start_date:
            b_filtered = b_filtered[b_filtered.index >= pd.Timestamp(start_date)]
        if end_date:
            b_filtered = b_filtered[b_filtered.index <= pd.Timestamp(end_date)]
        b_close = b_filtered["Close"].dropna()
        if not b_close.empty:
            bench_equities[b_name] = cfg.backtest.initial_capital * (b_close / b_close.iloc[0])
    bench_equities["cash_plus_3"] = _cash_hurdle_equity(
        equity, cfg.backtest.initial_capital, rf_annual, cfg.backtest.abs_return_hurdle
    )

    metrics = compute_all_metrics(
        equity=equity,
        initial_capital=cfg.backtest.initial_capital,
        trades_df=trades_df,
        benchmarks=bench_equities,
        rf_annual=rf_annual,
    )

    return BacktestResult(
        ticker=ticker,
        equity_curve=equity,
        trades_df=trades_df,
        metrics=metrics,
        benchmark_equity=bench_equities,
    )


def run_backtest(
    cfg: TrendFollowingConfig,
    ticker_ohlc: Dict[str, pd.DataFrame],
    benchmark_ohlc: Dict[str, pd.DataFrame],
    rf_annual: float = 0.04,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> BacktestSummary:
    """Run the trend-following backtest across all tickers.

    Args:
        cfg: Strategy configuration.
        ticker_ohlc: Dict of ticker -> OHLCV DataFrame.
        benchmark_ohlc: Dict of benchmark_name -> OHLCV DataFrame.
        rf_annual: Annualised risk-free rate.
        start_date: ISO date string for backtest start (optional).
        end_date: ISO date string for backtest end (optional).

    Returns:
        BacktestSummary with per-ticker results and combined portfolio metrics.
    """
    summary = BacktestSummary()

    for ticker, ohlc in ticker_ohlc.items():
        sentiment_hist = _load_sentiment_history(ticker)
        risk_hist = _load_risk_history(ticker)

        result = _run_single_ticker(
            ticker=ticker,
            ohlc=ohlc,
            cfg=cfg,
            sentiment_hist=sentiment_hist,
            risk_hist=risk_hist,
            benchmark_ohlc=benchmark_ohlc,
            rf_annual=rf_annual,
            start_date=start_date,
            end_date=end_date,
        )
        if result is not None:
            summary.results[ticker] = result

    # Combined portfolio: equal-weight average of normalised equity curves
    valid_curves = [
        r.equity_curve for r in summary.results.values() if not r.equity_curve.empty
    ]
    if valid_curves:
        normalised = [c / c.iloc[0] for c in valid_curves]
        combined_norm = pd.concat(normalised, axis=1).ffill().mean(axis=1)
        combined_equity = cfg.backtest.initial_capital * combined_norm
        summary.portfolio_equity = combined_equity

        bench_equities: Dict[str, pd.Series] = {}
        for b_name, b_ohlc in benchmark_ohlc.items():
            b_filtered = b_ohlc.copy()
            if start_date:
                b_filtered = b_filtered[b_filtered.index >= pd.Timestamp(start_date)]
            if end_date:
                b_filtered = b_filtered[b_filtered.index <= pd.Timestamp(end_date)]
            b_close = b_filtered["Close"].dropna()
            if not b_close.empty:
                bench_equities[b_name] = (
                    cfg.backtest.initial_capital * (b_close / b_close.iloc[0])
                )
        bench_equities["cash_plus_3"] = _cash_hurdle_equity(
            combined_equity, cfg.backtest.initial_capital, rf_annual, cfg.backtest.abs_return_hurdle
        )

        all_trades: List[pd.DataFrame] = [
            r.trades_df for r in summary.results.values() if not r.trades_df.empty
        ]
        combined_trades = (
            pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        )

        summary.portfolio_metrics = compute_all_metrics(
            equity=combined_equity,
            initial_capital=cfg.backtest.initial_capital,
            trades_df=combined_trades,
            benchmarks=bench_equities,
            rf_annual=rf_annual,
        )

    return summary


# ── Portfolio backtest helpers ─────────────────────────────────────────────────

def _precompute_indicators(
    ticker: str,
    ohlc: pd.DataFrame,
    cfg: "TrendFollowingConfig",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[Dict]:
    """Pre-compute all indicator series for one ticker.

    Returns None when the ticker has insufficient data after date filtering.
    Used exclusively by run_portfolio_backtest().
    """
    if start_date:
        ohlc = ohlc[ohlc.index >= pd.Timestamp(start_date)]
    if end_date:
        ohlc = ohlc[ohlc.index <= pd.Timestamp(end_date)]

    ind_cfg = cfg.get_indicator_cfg(ticker)
    min_bars = max(
        ind_cfg.slow_period, cfg.adx.period * 2,
        cfg.rsi.period, cfg.macd.slow + cfg.macd.signal,
        cfg.volume.period, cfg.atr_stop.period,
        cfg.range_filter.lookback_days if cfg.range_filter.enabled else 0,
    ) + 10
    if ohlc.empty or len(ohlc) < min_bars:
        return None

    if cfg.macd.use_macd_entry:
        ema_gate = EMAIndicator(cfg.macd.trend_gate_period).compute(ohlc).values
        macd_prim = MACD(
            fast=cfg.macd.fast, slow=cfg.macd.slow, signal=cfg.macd.signal
        ).compute(ohlc).values
        close_s = ohlc["Close"]
        valid = (~ema_gate.isna()) & (~macd_prim.isna())
        above = close_s > ema_gate
        direction = pd.Series(np.nan, index=ohlc.index)
        direction[valid & above & (macd_prim > 0)] = 1.0
        direction[valid & (~above | (macd_prim <= 0))] = -1.0
    else:
        direction = CrossoverSignal(
            fast_window=ind_cfg.fast_period,
            slow_window=ind_cfg.slow_period,
            ma_type=ind_cfg.ma_type,
        ).signal_series(ohlc)

    adx = (
        ADX(period=cfg.adx.period, threshold=cfg.adx.min_adx).compute(ohlc).values
        if cfg.adx.enabled else None
    )
    rsi = (
        RSI(period=cfg.rsi.period, overbought=cfg.rsi.overbought).compute(ohlc).values
        if cfg.rsi.enabled else None
    )
    macd = (
        MACD(fast=cfg.macd.fast, slow=cfg.macd.slow, signal=cfg.macd.signal).compute(ohlc).values
        if cfg.macd.enabled else None
    )
    vol_ratio = (
        VolumeConfirmation(period=cfg.volume.period, min_ratio=cfg.volume.min_ratio)
        .compute(ohlc).raw["ratio"]
        if cfg.volume.enabled else None
    )
    range_pos = (
        RangeFilter(
            lookback_days=cfg.range_filter.lookback_days,
            top_block_threshold=cfg.range_filter.top_block_pct,
        ).compute(ohlc).values
        if cfg.range_filter.enabled else None
    )
    mtf = (
        MultiTimeframeConfirmation(
            fast_weeks=cfg.mtf.fast_weeks, slow_weeks=cfg.mtf.slow_weeks
        ).signal_series(ohlc)
        if cfg.mtf.enabled else None
    )
    vol_regime = (
        VolatilityRegime(
            period=cfg.vol_regime.period,
            low_vol_threshold=cfg.vol_regime.low_vol_threshold,
            high_vol_threshold=cfg.vol_regime.high_vol_threshold,
            min_multiplier=cfg.vol_regime.min_multiplier,
        ).signal_series(ohlc)
        if cfg.vol_regime.enabled else None
    )
    atr = (
        ATR(period=cfg.atr_stop.period).atr_series(ohlc)
        if cfg.atr_stop.enabled else None
    )
    return {
        "ticker": ticker,
        "ohlc": ohlc,
        "direction": direction,
        "adx": adx,
        "rsi": rsi,
        "macd": macd,
        "vol_ratio": vol_ratio,
        "range_pos": range_pos,
        "mtf": mtf,
        "vol_regime": vol_regime,
        "atr": atr,
        "sentiment_hist": _load_sentiment_history(ticker),
        "risk_hist": _load_risk_history(ticker),
    }


def run_portfolio_backtest(
    cfg: "TrendFollowingConfig",
    ticker_ohlc: Dict[str, pd.DataFrame],
    benchmark_ohlc: Dict[str, pd.DataFrame],
    rf_annual: float = 0.04,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> "BacktestSummary":
    """Multi-ticker portfolio backtest with portfolio-level constraints.

    Unlike run_backtest() (which runs each ticker in isolation with independent
    capital), this function uses a **single shared portfolio** and enforces:

      - max_open_positions  : cap on simultaneous open trades
      - max_sector_exposure_pct : sector concentration limit (% of NAV)
      - max_gross_exposure_pct  : max (long + short) notional / NAV
      - adv_participation_pct   : volume-based order size cap (via shares_to_buy)

    All constraints are configured via cfg.portfolio_constraints.
    """
    pc = cfg.portfolio_constraints
    sm = cfg.sector_map

    # ── Pre-compute all indicators ─────────────────────────────────────────────
    all_ind: Dict[str, Dict] = {}
    for ticker, ohlc in ticker_ohlc.items():
        ind = _precompute_indicators(ticker, ohlc, cfg, start_date, end_date)
        if ind is not None:
            all_ind[ticker] = ind

    if not all_ind:
        return BacktestSummary()

    # ── Unified trading calendar (sorted union of all valid dates) ─────────────
    all_dates: List = sorted(set().union(*[ind["direction"].dropna().index for ind in all_ind.values()]))

    # ── Shared portfolio ───────────────────────────────────────────────────────
    portfolio = Portfolio(cfg.backtest.initial_capital, cfg.backtest.commission_pct)
    slippage = cfg.backtest.slippage
    daily_rf = (1.0 + rf_annual) ** (1.0 / 252) - 1.0

    atr_stops: Dict[str, float] = {}
    peak_prices: Dict[str, float] = {}
    short_stops: Dict[str, float] = {}
    last_prices: Dict[str, float] = {}  # forward-filled close prices

    for dt in all_dates:
        dt_date = dt.date()
        date_str = dt_date.isoformat()

        # Forward-fill latest known prices
        for ticker, ind in all_ind.items():
            if dt in ind["ohlc"].index:
                last_prices[ticker] = float(ind["ohlc"].loc[dt, "Close"])

        if cfg.backtest.model_cash_interest:
            portfolio.accrue_cash_interest(daily_rf)

        # ── Exits / stop checks ────────────────────────────────────────────────
        for ticker, ind in all_ind.items():
            if dt not in ind["ohlc"].index:
                continue
            cp = last_prices.get(ticker, 0.0)
            if cp <= 0:
                continue

            if cfg.atr_stop.enabled and portfolio.is_invested(ticker):
                if ticker in atr_stops and cp <= atr_stops[ticker]:
                    portfolio.sell_all(ticker, cp * (1.0 - slippage), date_str)
                    atr_stops.pop(ticker, None)
                    peak_prices.pop(ticker, None)
                    continue
                if ticker in peak_prices:
                    peak_prices[ticker] = max(peak_prices[ticker], cp)
                if cfg.atr_stop.profit_stop_enabled and ticker in peak_prices:
                    atr_v = _val_at(ind["atr"], dt, 0.0)
                    if atr_v > 0 and cp <= peak_prices[ticker] - cfg.atr_stop.profit_stop_atr_mult * atr_v:
                        portfolio.sell_all(ticker, cp * (1.0 - slippage), date_str)
                        atr_stops.pop(ticker, None)
                        peak_prices.pop(ticker, None)
                        continue
                if cfg.atr_stop.trail:
                    atr_v = _val_at(ind["atr"], dt, 0.0)
                    if atr_v > 0 and ticker in atr_stops:
                        atr_stops[ticker] = max(atr_stops[ticker], cp - cfg.atr_stop.multiplier * atr_v)

            if cfg.atr_stop.enabled and portfolio.is_short(ticker):
                if ticker in short_stops and cp >= short_stops[ticker]:
                    portfolio.cover_all(ticker, cp * (1.0 + slippage), date_str)
                    short_stops.pop(ticker, None)
                    continue

        # ── Build filtered actions ─────────────────────────────────────────────
        exits: List = []
        entries: List = []

        for ticker, ind in all_ind.items():
            direction_val = _val_at(ind["direction"], dt)
            if direction_val is None:
                continue
            cp = last_prices.get(ticker, 0.0)
            if cp <= 0:
                continue

            sent_snap = _get_as_of(ind["sentiment_hist"], dt_date)
            risk_snap = _get_as_of(ind["risk_hist"], dt_date)
            overall_sent = (sent_snap or {}).get("overall_sentiment")
            conf = float((sent_snap or {}).get("confidence") or 0.0)
            risk_score = (risk_snap or {}).get("composite_risk_score")
            kelly_frac = (risk_snap or {}).get("kelly_fraction_capped")
            db_stop = (risk_snap or {}).get("suggested_stop_loss_pct")

            raw_action: Action = (
                "BUY" if direction_val > 0
                else ("SHORT" if cfg.short.enabled else "SELL") if direction_val < 0
                else "HOLD"
            )
            filtered_action, _ = apply_filters(
                raw_action=raw_action, last_direction=direction_val, cfg=cfg,
                adx_val=_val_at(ind["adx"], dt) if cfg.adx.enabled else None,
                rsi_val=_val_at(ind["rsi"], dt) if cfg.rsi.enabled else None,
                macd_hist=_val_at(ind["macd"], dt) if cfg.macd.enabled else None,
                vol_ratio=_val_at(ind["vol_ratio"], dt) if cfg.volume.enabled else None,
                range_pos=_val_at(ind["range_pos"], dt) if cfg.range_filter.enabled else None,
                weekly_trend=_val_at(ind["mtf"], dt) if cfg.mtf.enabled else None,
                sentiment_data=sent_snap, risk_data=risk_snap,
            )
            vol_mult = _val_at(ind["vol_regime"], dt, 1.0) if cfg.vol_regime.enabled else 1.0
            daily_vol = (
                float(ind["ohlc"].loc[dt, "Volume"])
                if "Volume" in ind["ohlc"].columns and dt in ind["ohlc"].index else None
            )
            sig = Signal(
                ticker=ticker, date=date_str, action=filtered_action,
                trend_direction=direction_val, filtered_strength=abs(direction_val) * vol_mult,
                reason="", sentiment=overall_sent,
                sentiment_confidence=conf if conf > 0 else None, risk_score=risk_score,
            )

            if filtered_action == "SELL" and portfolio.is_invested(ticker):
                exits.append((ticker, "SELL", cp * (1.0 - slippage)))
            elif filtered_action == "COVER" and portfolio.is_short(ticker):
                exits.append((ticker, "COVER", cp * (1.0 + slippage)))
            elif filtered_action == "BUY" and not portfolio.is_invested(ticker) and not portfolio.is_short(ticker):
                entries.append((ticker, "BUY", sig, cp * (1.0 + slippage), kelly_frac, db_stop, daily_vol, cp))
            elif filtered_action == "SHORT" and not portfolio.is_short(ticker) and not portfolio.is_invested(ticker):
                entries.append((ticker, "SHORT", sig, cp * (1.0 - slippage), kelly_frac, db_stop, daily_vol, cp))

        # ── Execute exits first ────────────────────────────────────────────────
        for ticker, action, exec_price in exits:
            if action == "SELL":
                portfolio.sell_all(ticker, exec_price, date_str)
                atr_stops.pop(ticker, None)
                peak_prices.pop(ticker, None)
            else:
                portfolio.cover_all(ticker, exec_price, date_str)
                short_stops.pop(ticker, None)

        # ── Apply portfolio constraints and execute entries ─────────────────────
        current_nav = portfolio.equity(last_prices)
        gross_exp = portfolio.gross_exposure(last_prices)
        sector_exp = portfolio.sector_exposure(last_prices, sm)
        open_count = portfolio.open_position_count()

        for ticker, action, sig, exec_price, kelly_frac, db_stop, daily_vol, cp in entries:
            if pc.max_open_positions > 0 and open_count >= pc.max_open_positions:
                continue
            if pc.max_gross_exposure_pct > 0 and current_nav > 0:
                if gross_exp / current_nav * 100.0 >= pc.max_gross_exposure_pct:
                    continue
            if pc.max_sector_exposure_pct > 0 and current_nav > 0:
                sector_pct = sector_exp.get(sm.get(ticker, "Unknown"), 0.0) / current_nav * 100.0
                if sector_pct >= pc.max_sector_exposure_pct:
                    continue

            n_shares = shares_to_buy(sig, current_nav, exec_price, cfg,
                                     kelly_fraction=kelly_frac, daily_volume=daily_vol)
            if n_shares <= 0:
                continue

            if action == "BUY":
                if portfolio.buy(ticker, n_shares, exec_price, date_str):
                    peak_prices[ticker] = exec_price
                    if cfg.atr_stop.enabled:
                        if db_stop is not None and cfg.atr_stop.use_db_stop_when_available:
                            atr_stops[ticker] = exec_price * (1.0 + db_stop)
                        else:
                            atr_v = _val_at(all_ind[ticker]["atr"], dt, 0.0)
                            if atr_v > 0:
                                atr_stops[ticker] = exec_price - cfg.atr_stop.multiplier * atr_v
                    gross_exp += n_shares * cp
                    sect = sm.get(ticker, "Unknown")
                    sector_exp[sect] = sector_exp.get(sect, 0.0) + n_shares * cp
                    open_count += 1
            else:  # SHORT
                if portfolio.short(ticker, n_shares, exec_price, date_str):
                    atr_v = _val_at(all_ind[ticker]["atr"], dt, 0.0)
                    if atr_v > 0:
                        short_stops[ticker] = exec_price + cfg.atr_stop.multiplier * atr_v
                    gross_exp += n_shares * cp
                    sect = sm.get(ticker, "Unknown")
                    sector_exp[sect] = sector_exp.get(sect, 0.0) + n_shares * cp
                    open_count += 1

        portfolio.record_equity(date_str, last_prices)

    # ── Build results ──────────────────────────────────────────────────────────
    all_trades_df = portfolio.to_trades_df()
    portfolio_equity = portfolio.equity_series()

    bench_equities: Dict[str, pd.Series] = {}
    for b_name, b_ohlc in benchmark_ohlc.items():
        b_f = b_ohlc.copy()
        if start_date:
            b_f = b_f[b_f.index >= pd.Timestamp(start_date)]
        if end_date:
            b_f = b_f[b_f.index <= pd.Timestamp(end_date)]
        b_close = b_f["Close"].dropna()
        if not b_close.empty:
            bench_equities[b_name] = cfg.backtest.initial_capital * (b_close / b_close.iloc[0])
    bench_equities["cash_plus_3"] = _cash_hurdle_equity(
        portfolio_equity, cfg.backtest.initial_capital, rf_annual, cfg.backtest.abs_return_hurdle
    )

    summary = BacktestSummary()
    summary.portfolio_equity = portfolio_equity
    summary.portfolio_metrics = compute_all_metrics(
        equity=portfolio_equity, initial_capital=cfg.backtest.initial_capital,
        trades_df=all_trades_df, benchmarks=bench_equities, rf_annual=rf_annual,
    )

    for ticker in all_ind:
        t_trades = (
            all_trades_df[all_trades_df["ticker"] == ticker].reset_index(drop=True)
            if not all_trades_df.empty else pd.DataFrame()
        )
        summary.results[ticker] = BacktestResult(
            ticker=ticker,
            equity_curve=portfolio_equity,
            trades_df=t_trades,
            metrics=compute_all_metrics(
                equity=portfolio_equity, initial_capital=cfg.backtest.initial_capital,
                trades_df=t_trades, benchmarks=bench_equities, rf_annual=rf_annual,
            ),
            benchmark_equity=bench_equities,
        )

    return summary
