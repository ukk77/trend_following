"""Signal generator — layered indicator + sentiment + risk pipeline.

Filter order (each layer can only demote BUY → HOLD, not promote):
    1.  EMA/SMA crossover         → raw direction (+1 / -1 / 0)
    2.  ADX filter                → HOLD if trend too weak
    3.  RSI filter                → HOLD on BUY if overbought
    4.  MACD confirmation         → HOLD on BUY if momentum disagrees
    5.  Volume confirmation       → HOLD if below-average volume
    6.  52-week range filter      → HOLD on BUY if price overextended
    7.  Multi-timeframe filter    → HOLD if weekly trend disagrees
    8.  Sentiment filter (DB)     → HOLD on BUY if negative/low-confidence
    9.  Risk filter (DB)          → HOLD on BUY if risk score too high
   10.  Volatility regime         → position-size multiplier (not a HOLD filter)
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd

from ..config import TrendFollowingConfig
from ..indicators.moving_averages import CrossoverSignal
from ..indicators.momentum import RSI, MACD
from ..indicators.trend_strength import ADX
from ..indicators.volatility import ATR, VolatilityRegime
from ..indicators.volume import VolumeConfirmation
from ..indicators.range_filter import RangeFilter, MultiTimeframeConfirmation

Action = Literal["BUY", "SELL", "HOLD"]

_TRADING_ROOT = Path(__file__).resolve().parents[2]
_SENTIMENT_DB = _TRADING_ROOT / "sentiment_analysis" / "backend" / "sentiment_history.db"
_RISK_DB = _TRADING_ROOT / "risk_calculator" / "backend" / "risk_history.db"


@dataclass
class Signal:
    """Output of the signal generator for one ticker on one date."""
    ticker: str
    date: str
    action: Action
    trend_direction: float
    filtered_strength: float
    reason: str
    sentiment: Optional[str] = None
    sentiment_confidence: Optional[float] = None
    risk_score: Optional[float] = None
    risk_bucket: Optional[str] = None
    # Indicator snapshot values (for reporting/logging)
    rsi_value: Optional[float] = None
    adx_value: Optional[float] = None
    macd_hist: Optional[float] = None
    volume_ratio: Optional[float] = None
    range_position: Optional[float] = None
    weekly_trend: Optional[float] = None
    vol_regime_mult: Optional[float] = None
    atr_stop: Optional[float] = None


def _fetch_latest_sentiment(ticker: str) -> Optional[dict]:
    """Look up the most recent sentiment snapshot from the history DB."""
    if not _SENTIMENT_DB.exists():
        return None
    try:
        with sqlite3.connect(str(_SENTIMENT_DB)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT overall_sentiment, confidence, avg_sentiment "
                "FROM sentiment_snapshots "
                "WHERE UPPER(ticker)=UPPER(?) ORDER BY captured_at DESC LIMIT 1",
                (ticker.upper(),),
            ).fetchone()
        return dict(row) if row else None
    except Exception:
        return None


def _fetch_latest_risk(ticker: str) -> Optional[dict]:
    """Look up the most recent risk snapshot from the history DB."""
    if not _RISK_DB.exists():
        return None
    try:
        with sqlite3.connect(str(_RISK_DB)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT composite_risk_score, risk_bucket, overall_sentiment, upstream_confidence "
                "FROM risk_snapshots "
                "WHERE UPPER(ticker)=UPPER(?) ORDER BY captured_at DESC LIMIT 1",
                (ticker.upper(),),
            ).fetchone()
        return dict(row) if row else None
    except Exception:
        return None


def generate_signal(
    ticker: str,
    ohlc: pd.DataFrame,
    cfg: TrendFollowingConfig,
    sentiment_override: Optional[dict] = None,
    risk_override: Optional[dict] = None,
) -> Signal:
    """Generate a trading signal for one ticker applying all 10 filter layers.

    Args:
        ticker: Stock symbol.
        ohlc: OHLCV DataFrame indexed by datetime.
        cfg: Strategy configuration.
        sentiment_override: Pre-loaded sentiment dict (skips DB lookup).
        risk_override: Pre-loaded risk dict (skips DB lookup).

    Returns:
        Signal with action, filtered strength, indicator snapshot, and reason chain.
    """
    today_str = datetime.now(timezone.utc).date().isoformat()
    reasons: List[str] = []

    # ── Layer 1: EMA/SMA crossover ────────────────────────────────────────────
    crossover = CrossoverSignal(
        fast_window=cfg.indicator.fast_period,
        slow_window=cfg.indicator.slow_period,
        ma_type=cfg.indicator.ma_type,
    )
    direction_series = crossover.signal_series(ohlc)
    clean = direction_series.dropna()
    last_direction = float(clean.iloc[-1]) if not clean.empty else 0.0
    reasons.append(f"trend={crossover.name} dir={last_direction:+.0f}")

    if last_direction > 0:
        filtered_action: Action = "BUY"
    elif last_direction < 0:
        filtered_action = "SELL"
    else:
        filtered_action = "HOLD"

    # ── Layer 2: ADX — only trade in trending markets ─────────────────────────
    adx_value: Optional[float] = None
    if cfg.adx.enabled and filtered_action != "HOLD":
        adx_ind = ADX(period=cfg.adx.period, threshold=cfg.adx.min_adx)
        adx_value = adx_ind.latest_value(ohlc)
        if not adx_ind.is_trending(ohlc):
            filtered_action = "HOLD"
            reasons.append(f"adx={adx_value:.1f}<{cfg.adx.min_adx}(no_trend)")
        else:
            reasons.append(f"adx={adx_value:.1f}OK")

    # ── Layer 3: RSI — block BUY when overbought ──────────────────────────────
    rsi_value: Optional[float] = None
    if cfg.rsi.enabled and filtered_action == "BUY":
        rsi_ind = RSI(
            period=cfg.rsi.period,
            overbought=cfg.rsi.overbought,
            oversold=cfg.rsi.oversold,
        )
        rsi_value = rsi_ind.latest_value(ohlc)
        if rsi_ind.is_overbought(ohlc):
            filtered_action = "HOLD"
            reasons.append(f"rsi={rsi_value:.1f}>={cfg.rsi.overbought}(overbought)")
        else:
            reasons.append(f"rsi={rsi_value:.1f}OK")

    # ── Layer 4: MACD — confirm momentum direction ────────────────────────────
    macd_hist: Optional[float] = None
    if cfg.macd.enabled and filtered_action == "BUY":
        macd_ind = MACD(fast=cfg.macd.fast, slow=cfg.macd.slow, signal=cfg.macd.signal)
        hist_series = macd_ind.compute(ohlc).values.dropna()
        macd_hist = float(hist_series.iloc[-1]) if not hist_series.empty else None
        if not macd_ind.histogram_confirms_buy(ohlc):
            filtered_action = "HOLD"
            reasons.append(f"macd_hist={macd_hist:.4f}(bearish)")
        else:
            reasons.append(f"macd_hist={macd_hist:.4f}OK")

    # ── Layer 5: Volume confirmation ──────────────────────────────────────────
    volume_ratio: Optional[float] = None
    if cfg.volume.enabled and filtered_action != "HOLD":
        vol_ind = VolumeConfirmation(period=cfg.volume.period, min_ratio=cfg.volume.min_ratio)
        volume_ratio = vol_ind.latest_ratio(ohlc)
        if not vol_ind.is_confirmed(ohlc):
            filtered_action = "HOLD"
            reasons.append(f"vol_ratio={volume_ratio:.2f}<{cfg.volume.min_ratio}(low_vol)")
        else:
            reasons.append(f"vol_ratio={volume_ratio:.2f}OK")

    # ── Layer 6: 52-week range filter — block BUY if overextended ────────────
    range_position: Optional[float] = None
    if cfg.range_filter.enabled and filtered_action == "BUY":
        rng_ind = RangeFilter(
            lookback_days=cfg.range_filter.lookback_days,
            top_block_threshold=cfg.range_filter.top_block_pct,
        )
        range_position = rng_ind.latest_position(ohlc)
        if rng_ind.is_overextended(ohlc):
            filtered_action = "HOLD"
            reasons.append(f"range={range_position:.2f}>{cfg.range_filter.top_block_pct}(overextended)")
        else:
            reasons.append(f"range={range_position:.2f}OK")

    # ── Layer 7: Multi-timeframe — weekly trend must agree ────────────────────
    weekly_trend: Optional[float] = None
    if cfg.mtf.enabled and filtered_action != "HOLD":
        mtf_ind = MultiTimeframeConfirmation(
            fast_weeks=cfg.mtf.fast_weeks,
            slow_weeks=cfg.mtf.slow_weeks,
        )
        weekly_trend = mtf_ind.weekly_trend(ohlc)
        if weekly_trend != 0 and weekly_trend != last_direction:
            filtered_action = "HOLD"
            reasons.append(f"mtf_weekly={weekly_trend:+.0f}(disagrees)")
        else:
            reasons.append(f"mtf_weekly={weekly_trend:+.0f}OK")

    # ── Layer 8: Sentiment filter (DB) ────────────────────────────────────────
    sentiment_data = sentiment_override or _fetch_latest_sentiment(ticker)
    risk_data = risk_override or _fetch_latest_risk(ticker)

    overall_sentiment = (sentiment_data or {}).get("overall_sentiment")
    conf = float((sentiment_data or {}).get("confidence") or 0.0)
    risk_score = (risk_data or {}).get("composite_risk_score")
    risk_bucket = (risk_data or {}).get("risk_bucket")

    if filtered_action == "BUY" and cfg.signal.sentiment_filter_enabled:
        if conf < cfg.signal.min_sentiment_confidence:
            filtered_action = "HOLD"
            reasons.append(f"low_conf={conf:.2f}<{cfg.signal.min_sentiment_confidence}")
        elif cfg.signal.block_on_negative_sentiment and overall_sentiment == "negative":
            filtered_action = "HOLD"
            reasons.append("blocked:negative_sentiment")

    # ── Layer 9: Risk filter (DB) ─────────────────────────────────────────────
    if filtered_action == "BUY" and cfg.signal.risk_filter_enabled and risk_score is not None:
        if risk_score > cfg.signal.max_risk_score:
            filtered_action = "HOLD"
            reasons.append(f"risk={risk_score:.1f}>{cfg.signal.max_risk_score}")

    # ── Layer 10: Volatility regime — position-size multiplier ───────────────
    vol_regime_mult: Optional[float] = None
    if cfg.vol_regime.enabled:
        vr = VolatilityRegime(
            period=cfg.vol_regime.period,
            low_vol_threshold=cfg.vol_regime.low_vol_threshold,
            high_vol_threshold=cfg.vol_regime.high_vol_threshold,
            min_multiplier=cfg.vol_regime.min_multiplier,
        )
        vol_regime_mult = vr.latest_multiplier(ohlc)
        if vol_regime_mult < 1.0:
            reasons.append(f"vol_regime_mult={vol_regime_mult:.2f}")

    # ── ATR stop price (informational) ───────────────────────────────────────
    atr_stop: Optional[float] = None
    if cfg.atr_stop.enabled:
        atr_ind = ATR(period=cfg.atr_stop.period)
        current_price = float(ohlc["Close"].iloc[-1])
        atr_stop = atr_ind.stop_price(
            ohlc, current_price, cfg.atr_stop.multiplier, direction="long"
        )

    # ── Compute final position strength ──────────────────────────────────────
    if filtered_action == "HOLD":
        strength = 0.0
    elif filtered_action == "BUY":
        ps = cfg.position_sizing
        if overall_sentiment == "positive" and conf >= cfg.signal.min_sentiment_confidence:
            sent_mult = ps.sentiment_agree_mult
            reasons.append(f"sent=positive(x{sent_mult})")
        elif overall_sentiment == "negative":
            sent_mult = ps.sentiment_disagree_mult
            reasons.append(f"sent=negative(x{sent_mult})")
        else:
            sent_mult = ps.sentiment_neutral_mult
            reasons.append(f"sent=neutral(x{sent_mult})")
        strength = min(abs(last_direction) * sent_mult * (vol_regime_mult or 1.0), 1.0)
    else:  # SELL
        strength = abs(last_direction)

    return Signal(
        ticker=ticker,
        date=today_str,
        action=filtered_action,
        trend_direction=last_direction,
        filtered_strength=strength,
        reason=" | ".join(reasons),
        sentiment=overall_sentiment,
        sentiment_confidence=conf if conf > 0 else None,
        risk_score=risk_score,
        risk_bucket=risk_bucket,
        rsi_value=rsi_value,
        adx_value=adx_value,
        macd_hist=macd_hist,
        volume_ratio=volume_ratio,
        range_position=range_position,
        weekly_trend=weekly_trend,
        vol_regime_mult=vol_regime_mult,
        atr_stop=atr_stop,
    )


def generate_all_signals(
    ohlc_map: Dict[str, pd.DataFrame],
    cfg: TrendFollowingConfig,
) -> List[Signal]:
    """Generate signals for all tickers in cfg.tickers."""
    signals = []
    for ticker in cfg.tickers:
        if ticker not in ohlc_map:
            continue
        sig = generate_signal(ticker, ohlc_map[ticker], cfg)
        signals.append(sig)
    return signals
