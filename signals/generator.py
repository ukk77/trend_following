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

import os
import requests
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd

from ..config import TrendFollowingConfig
from ..indicators.moving_averages import CrossoverSignal, EMA as EMAIndicator
from ..indicators.momentum import RSI, MACD
from ..indicators.trend_strength import ADX
from ..indicators.volatility import ATR, VolatilityRegime
from ..indicators.volume import VolumeConfirmation
from ..indicators.range_filter import RangeFilter, MultiTimeframeConfirmation
from .filters import apply_filters, Action

try:
    from trading_core.session_context import (
        fetch_premarket_gap,
        premarket_confirmation_mult,
        early_session_size_scalar,
    )
except ImportError:
    def fetch_premarket_gap(ticker): return None  # type: ignore
    def premarket_confirmation_mult(gap, direction, **kw): return 1.0  # type: ignore
    def early_session_size_scalar(**kw): return 1.0  # type: ignore

try:
    from trading_core.regime_params import get_regime_adjustments, RegimeAdjustments
except ImportError:
    def get_regime_adjustments(strategy, **kw):  # type: ignore
        from dataclasses import dataclass
        @dataclass
        class _Neutral:
            regime: str = "range_bound"
            regime_detected: bool = False
            position_size_mult: float = 1.0
            entry_threshold_mult: float = 1.0
            stop_width_mult: float = 1.0
            adx_threshold_mult: float = 1.0
            sentiment_strictness_mult: float = 1.0
            max_risk_score_mult: float = 1.0
        return _Neutral()

_TRADING_ROOT = Path(__file__).resolve().parents[2]
_SENTIMENT_DB = Path(os.getenv("SENTIMENT_DB_PATH",
    str(_TRADING_ROOT / "sentiment_analysis" / "backend" / "sentiment_history.db")))
_RISK_DB = Path(os.getenv("RISK_DB_PATH",
    str(_TRADING_ROOT / "risk_calculator" / "backend" / "risk_history.db")))


@dataclass
class Signal:
    """Output of the signal generator for one ticker on one date."""
    ticker: str
    date: str
    action: Action
    trend_direction: float
    filtered_strength: float
    raw_strength: float
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


import logging
log = logging.getLogger(__name__)

def _fetch_latest_sentiment(ticker: str) -> Optional[dict]:
    """Look up the most recent sentiment snapshot - DB-first, API fallback."""
    import sqlite3
    if _SENTIMENT_DB.exists():
        try:
            with sqlite3.connect(str(_SENTIMENT_DB)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT overall_sentiment, confidence, avg_sentiment "
                    "FROM sentiment_snapshots WHERE UPPER(ticker)=UPPER(?) "
                    "ORDER BY captured_at DESC LIMIT 1",
                    (ticker,)
                ).fetchone()
                if row:
                    return dict(row)
        except Exception:
            pass
    url = os.getenv("SENTIMENT_API_URL", "http://localhost:8000")
    try:
        resp = requests.get(f"{url}/api/history/{ticker}?limit=1", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("snapshots") and len(data["snapshots"]) > 0:
                return data["snapshots"][0]
        else:
            log.warning("fetch_sentiment %s returned status %s", ticker, resp.status_code)
    except Exception as e:
        log.warning("fetch_sentiment %s failed: %s", ticker, type(e).__name__)
    return None


def _fetch_latest_risk(ticker: str) -> Optional[dict]:
    """Look up the most recent risk snapshot - DB-first, API fallback."""
    import sqlite3
    if _RISK_DB.exists():
        try:
            with sqlite3.connect(str(_RISK_DB)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT composite_risk_score, risk_bucket, kelly_fraction_capped "
                    "FROM risk_snapshots WHERE UPPER(ticker)=UPPER(?) "
                    "ORDER BY captured_at DESC LIMIT 1",
                    (ticker,)
                ).fetchone()
                if row:
                    return dict(row)
        except Exception:
            pass
    url = os.getenv("RISK_API_URL", "http://localhost:8100")
    try:
        resp = requests.get(f"{url}/api/history/{ticker}?limit=1", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("snapshots") and len(data["snapshots"]) > 0:
                return data["snapshots"][0]
        else:
            log.warning("fetch_risk %s returned status %s", ticker, resp.status_code)
    except Exception as e:
        log.warning("fetch_risk %s failed: %s", ticker, type(e).__name__)
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

    # ── Regime-adaptive adjustments ─────────────────────────────────────────
    regime_adj = get_regime_adjustments("trend_following")
    if regime_adj.regime_detected:
        reasons.append(f"regime={regime_adj.regime}")

    # ── Layer 1: Primary trend direction (continuous strength) ──────────────
    ind_cfg = cfg.get_indicator_cfg(ticker)
    trend_intensity = 0.0  # continuous measure of trend strength: >0 bullish, <0 bearish
    if cfg.macd.use_macd_entry:
        ema_gate = EMAIndicator(cfg.macd.trend_gate_period).compute(ohlc).values
        macd_ind = MACD(fast=cfg.macd.fast, slow=cfg.macd.slow, signal=cfg.macd.signal)
        macd_hist_vals = macd_ind.compute(ohlc).values
        latest_close = float(ohlc["Close"].iloc[-1])
        latest_ema = float(ema_gate.dropna().iloc[-1]) if not ema_gate.dropna().empty else 0.0
        latest_macd = float(macd_hist_vals.dropna().iloc[-1]) if not macd_hist_vals.dropna().empty else 0.0
        above_gate = latest_close > latest_ema
        last_direction = 1.0 if (above_gate and latest_macd > 0) else -1.0
        # Continuous intensity: normalized MACD histogram relative to price
        price_for_norm = latest_close if latest_close > 0 else 1.0
        trend_intensity = latest_macd / price_for_norm * 100  # expressed as bps-like scale
        signal_name = f"MACD({cfg.macd.fast},{cfg.macd.slow},{cfg.macd.signal})+EMA{cfg.macd.trend_gate_period}gate"
        reasons.append(f"trend={signal_name} dir={last_direction:+.0f} intensity={trend_intensity:+.3f}")
    else:
        crossover = CrossoverSignal(
            fast_window=ind_cfg.fast_period,
            slow_window=ind_cfg.slow_period,
            ma_type=ind_cfg.ma_type,
        )
        result = crossover.compute(ohlc)
        direction_series = result.values
        clean = direction_series.dropna()
        last_direction = float(clean.iloc[-1]) if not clean.empty else 0.0
        # Use gap_pct as continuous trend intensity
        if result.raw is not None and "gap_pct" in result.raw.columns:
            gap_series = result.raw["gap_pct"].dropna()
            trend_intensity = float(gap_series.iloc[-1]) if not gap_series.empty else 0.0
        reasons.append(f"trend={crossover.name} dir={last_direction:+.0f} gap={trend_intensity:+.4f}")

    if last_direction > 0:
        filtered_action: Action = "BUY"
    elif last_direction < 0:
        filtered_action = "SHORT" if cfg.short.enabled else "SELL"
    else:
        filtered_action = "HOLD"

    # ── Compute indicator snapshot values ────────────────────────────────────
    effective_adx_min = cfg.adx.min_adx * regime_adj.adx_threshold_mult
    adx_value: Optional[float] = None
    if cfg.adx.enabled:
        adx_ind = ADX(period=cfg.adx.period, threshold=effective_adx_min)
        adx_value = adx_ind.latest_value(ohlc)

    rsi_value: Optional[float] = None
    if cfg.rsi.enabled:
        rsi_ind = RSI(period=cfg.rsi.period, overbought=cfg.rsi.overbought, oversold=cfg.rsi.oversold)
        rsi_value = rsi_ind.latest_value(ohlc)

    macd_hist: Optional[float] = None
    macd_bearish_div: bool = False
    if cfg.macd.enabled:
        macd_ind = MACD(fast=cfg.macd.fast, slow=cfg.macd.slow, signal=cfg.macd.signal)
        hist_series = macd_ind.compute(ohlc).values.dropna()
        macd_hist = float(hist_series.iloc[-1]) if not hist_series.empty else None
        if getattr(cfg.macd, 'divergence_filter', False) or getattr(cfg.macd, 'divergence_exit', False):
            if hasattr(macd_ind, 'check_bearish_divergence'):
                macd_bearish_div = macd_ind.check_bearish_divergence(ohlc)

    volume_ratio: Optional[float] = None
    if cfg.volume.enabled:
        vol_ind = VolumeConfirmation(period=cfg.volume.period, min_ratio=cfg.volume.min_ratio)
        volume_ratio = vol_ind.latest_ratio(ohlc)

    range_position: Optional[float] = None
    if cfg.range_filter.enabled:
        rng_ind = RangeFilter(lookback_days=cfg.range_filter.lookback_days, top_block_threshold=cfg.range_filter.top_block_pct)
        range_position = rng_ind.latest_position(ohlc)

    weekly_trend: Optional[float] = None
    if cfg.mtf.enabled:
        mtf_ind = MultiTimeframeConfirmation(fast_weeks=cfg.mtf.fast_weeks, slow_weeks=cfg.mtf.slow_weeks)
        weekly_trend = mtf_ind.weekly_trend(ohlc)

    # ── Layer 2-9: Shared filter pipeline ──────────────────────────────────
    sentiment_data = sentiment_override or _fetch_latest_sentiment(ticker)
    risk_data = risk_override or _fetch_latest_risk(ticker)

    filtered_action, filter_reasons = apply_filters(
        raw_action=filtered_action,
        last_direction=last_direction,
        cfg=cfg,
        adx_val=adx_value,
        rsi_val=rsi_value,
        macd_hist=macd_hist,
        macd_bearish_div=macd_bearish_div,
        vol_ratio=volume_ratio,
        range_pos=range_position,
        weekly_trend=weekly_trend,
        sentiment_data=sentiment_data,
        risk_data=risk_data,
    )
    reasons.extend(filter_reasons)

    # ── Extract sentiment/risk fields for strength calc ───────────────────
    overall_sentiment = (sentiment_data or {}).get("overall_sentiment")
    conf = float((sentiment_data or {}).get("confidence") or 0.0)
    contrarian_signal = (sentiment_data or {}).get("contrarian_signal")
    relative_sentiment = (sentiment_data or {}).get("relative_sentiment")
    risk_score = (risk_data or {}).get("composite_risk_score")
    risk_bucket = (risk_data or {}).get("risk_bucket")

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

    # ── ATR stop price (informational) ──────────────────────────────────────
    atr_stop: Optional[float] = None
    if cfg.atr_stop.enabled:
        atr_ind = ATR(period=cfg.atr_stop.period)
        current_price = float(ohlc["Close"].iloc[-1])
        atr_stop = atr_ind.stop_price(
            ohlc, current_price, cfg.atr_stop.multiplier, direction="long"
        )

    # ── Session context: pre-market gap confirmation + early-session scaling ──
    pm_gap = fetch_premarket_gap(ticker)
    pm_mult = premarket_confirmation_mult(pm_gap, filtered_action)
    es_scalar = early_session_size_scalar()
    if pm_gap is not None and abs(pm_gap) >= 0.005:
        reasons.append(f"pm_gap={pm_gap:+.1%}(x{pm_mult:.2f})")
    if es_scalar < 1.0:
        reasons.append(f"early_session(x{es_scalar:.2f})")

    # ── Compute final position strength ───────────────────────────────────────
    # Combined trend strength from ADX magnitude + MA gap (continuous signal).
    # ADX component: 0.5 at the minimum threshold → 1.0 at 2× threshold.
    # Intensity component: normalised MA gap (0.5% gap → 0.5, 2% gap → 1.0).
    if cfg.adx.enabled and adx_value is not None:
        adx_strength = min(adx_value / (2.0 * effective_adx_min), 1.0)
        reasons.append(f"adx_strength={adx_strength:.2f}(adx={adx_value:.1f})")
    else:
        adx_strength = abs(last_direction)

    # Continuous intensity from MA gap or MACD normalised histogram
    intensity_component = min(abs(trend_intensity) / 0.02, 1.0)  # 2% gap = full strength
    # Blend: 60% ADX + 40% intensity (ADX still dominates but gap adds nuance)
    combined_trend_strength = 0.6 * adx_strength + 0.4 * intensity_component
    reasons.append(f"trend_strength={combined_trend_strength:.2f}(adx={adx_strength:.2f},gap={intensity_component:.2f})")

    if filtered_action == "HOLD":
        strength = 0.0
        raw_str = 0.0
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
        
        # Contrarian signal adjustments for trend following
        contrarian_mult = 1.0
        if contrarian_signal == "extreme_bearish_opportunity":
            # Fear often marks the start of a new uptrend - enhance position
            contrarian_mult = 1.2
            reasons.append(f"contrarian=opp(x{contrarian_mult})")
        elif contrarian_signal == "extreme_bullish_caution":
            # Already filtered out, but if it somehow passed, reduce size
            contrarian_mult = 0.8
            reasons.append(f"contrarian=caution(x{contrarian_mult})")
        
        # Sector-relative sentiment adjustment
        sector_mult = 1.0
        if relative_sentiment is not None:
            if relative_sentiment > 0.15:  # Outperforming sector = tailwind
                sector_mult = 1.1
                reasons.append(f"sector=outperform(x{sector_mult})")
            elif relative_sentiment < -0.15:  # Underperforming sector = headwind
                sector_mult = 0.9
                reasons.append(f"sector=underperform(x{sector_mult})")
        
        raw_str = combined_trend_strength * sent_mult * contrarian_mult * sector_mult * (vol_regime_mult or 1.0) * pm_mult * es_scalar * regime_adj.position_size_mult
        strength = min(raw_str, 1.0)
    elif filtered_action == "SHORT":
        ps = cfg.position_sizing
        if overall_sentiment == "negative" and conf >= cfg.signal.min_sentiment_confidence:
            sent_mult = ps.sentiment_agree_mult
            reasons.append(f"sent=negative(x{sent_mult})")
        elif overall_sentiment == "positive":
            sent_mult = ps.sentiment_disagree_mult
            reasons.append(f"sent=positive(x{sent_mult})")
        else:
            sent_mult = ps.sentiment_neutral_mult
            reasons.append(f"sent=neutral(x{sent_mult})")
        
        # Contrarian adjustments for short positions
        contrarian_mult = 1.0
        if contrarian_signal == "extreme_bullish_caution":
            # Crowded long = good short opportunity
            contrarian_mult = 1.2
            reasons.append(f"contrarian=caution_short(x{contrarian_mult})")
        
        # Sector-relative adjustment for shorts
        sector_mult = 1.0
        if relative_sentiment is not None:
            if relative_sentiment < -0.15:  # Underperforming sector = good short
                sector_mult = 1.1
                reasons.append(f"sector=underperform_short(x{sector_mult})")
            elif relative_sentiment > 0.15:  # Outperforming = avoid short
                sector_mult = 0.9
                reasons.append(f"sector=outperform_avoid(x{sector_mult})")
        
        raw_str = combined_trend_strength * sent_mult * contrarian_mult * sector_mult * (vol_regime_mult or 1.0) * pm_mult * es_scalar * regime_adj.position_size_mult
        strength = min(raw_str, 1.0)
    else:  # SELL / COVER
        raw_str = combined_trend_strength
        strength = combined_trend_strength

    return Signal(
        ticker=ticker,
        date=today_str,
        action=filtered_action,
        trend_direction=last_direction,
        filtered_strength=strength,
        raw_strength=raw_str,
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
