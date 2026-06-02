"""Shared filter pipeline — used by both signal generator and backtest engine.

Accepts pre-resolved indicator values (not raw series) so both callers can
supply values however they compute them (live vs point-in-time).
"""
from __future__ import annotations

from typing import List, Literal, Optional

from ..config import TrendFollowingConfig

Action = Literal["BUY", "SELL", "HOLD", "SHORT", "COVER"]


def apply_filters(
    raw_action: Action,
    last_direction: float,
    cfg: TrendFollowingConfig,
    adx_val: Optional[float] = None,
    rsi_val: Optional[float] = None,
    macd_hist: Optional[float] = None,
    macd_bearish_div: bool = False,
    vol_ratio: Optional[float] = None,
    range_pos: Optional[float] = None,
    weekly_trend: Optional[float] = None,
    sentiment_data: Optional[dict] = None,
    risk_data: Optional[dict] = None,
) -> tuple[Action, List[str]]:
    """Apply all active filters to a raw action, returning (filtered_action, reasons).

    Each filter can only demote BUY/SELL → HOLD, never promote.
    """
    reasons: List[str] = []
    filtered = raw_action

    # ── ADX — only trade in trending markets ─────────────────────────────
    if cfg.adx.enabled and filtered in ("BUY", "SHORT"):
        if adx_val is not None and adx_val < cfg.adx.min_adx:
            filtered = "HOLD"
            reasons.append(f"adx={adx_val:.1f}<{cfg.adx.min_adx}(no_trend)")
        elif adx_val is not None:
            reasons.append(f"adx={adx_val:.1f}OK")

    # ── RSI — block BUY when overbought, block SHORT when not oversold ───
    if cfg.rsi.enabled and filtered == "BUY":
        if rsi_val is not None and rsi_val >= cfg.rsi.overbought:
            filtered = "HOLD"
            reasons.append(f"rsi={rsi_val:.1f}>={cfg.rsi.overbought}(overbought)")
        elif rsi_val is not None:
            reasons.append(f"rsi={rsi_val:.1f}OK")
    elif cfg.short.rsi_oversold_required and filtered == "SHORT":
        if rsi_val is not None and rsi_val >= cfg.rsi.oversold:
            filtered = "HOLD"
            reasons.append(f"rsi={rsi_val:.1f}>={cfg.rsi.oversold}(not_oversold)")
        elif rsi_val is not None:
            reasons.append(f"rsi={rsi_val:.1f}OK(oversold)")

    # ── MACD ─ confirm momentum direction (skipped when MACD is primary signal)
    if getattr(cfg.macd, 'divergence_filter', False) and macd_bearish_div and filtered == "BUY":
        filtered = "HOLD"
        reasons.append("macd_div=bearish(exhaustion)")
    elif getattr(cfg.macd, 'divergence_exit', False) and macd_bearish_div and filtered == "HOLD":
        # Force a SELL if holding and bearish divergence appears
        filtered = "SELL"
        reasons.append("macd_div_exit")
        
    if cfg.macd.enabled and not getattr(cfg.macd, 'use_macd_entry', False) and filtered == "BUY":
        if macd_hist is not None and macd_hist <= 0:
            filtered = "HOLD"
            reasons.append(f"macd_hist={macd_hist:.4f}(bearish)")
        elif macd_hist is not None:
            reasons.append(f"macd_hist={macd_hist:.4f}OK")
    elif cfg.macd.enabled and not cfg.macd.use_macd_entry and filtered == "SHORT":
        if macd_hist is not None and macd_hist >= 0:
            filtered = "HOLD"
            reasons.append(f"macd_hist={macd_hist:.4f}(bullish)")
        elif macd_hist is not None:
            reasons.append(f"macd_hist={macd_hist:.4f}OK")

    # ── Volume confirmation ──────────────────────────────────────────────
    if cfg.volume.enabled and filtered in ("BUY", "SHORT"):
        if vol_ratio is not None and vol_ratio < cfg.volume.min_ratio:
            filtered = "HOLD"
            reasons.append(f"vol_ratio={vol_ratio:.2f}<{cfg.volume.min_ratio}(low_vol)")
        elif vol_ratio is not None:
            reasons.append(f"vol_ratio={vol_ratio:.2f}OK")

    # ── 52-week range filter — block BUY if overextended ────────────────
    if cfg.range_filter.enabled and filtered == "BUY":
        if range_pos is not None and range_pos > cfg.range_filter.top_block_pct:
            filtered = "HOLD"
            reasons.append(f"range={range_pos:.2f}>{cfg.range_filter.top_block_pct}(overextended)")
        elif range_pos is not None:
            reasons.append(f"range={range_pos:.2f}OK")

    # ── Multi-timeframe — weekly trend must agree ────────────────────────
    if cfg.mtf.enabled and filtered in ("BUY", "SHORT"):
        if weekly_trend is not None and weekly_trend != 0 and weekly_trend != last_direction:
            filtered = "HOLD"
            reasons.append(f"mtf_weekly={weekly_trend:+.0f}(disagrees)")
        elif weekly_trend is not None:
            reasons.append(f"mtf_weekly={weekly_trend:+.0f}OK")

    # ── Sentiment filter (DB) ────────────────────────────────────────────
    overall_sentiment = (sentiment_data or {}).get("overall_sentiment")
    conf = float((sentiment_data or {}).get("confidence") or 0.0)

    if filtered == "BUY" and cfg.signal.sentiment_filter_enabled and sentiment_data is not None:
        if conf < cfg.signal.min_sentiment_confidence:
            filtered = "HOLD"
            reasons.append(f"low_conf={conf:.2f}<{cfg.signal.min_sentiment_confidence}")
        elif cfg.signal.block_on_negative_sentiment and overall_sentiment == "negative":
            filtered = "HOLD"
            reasons.append("blocked:negative_sentiment")
    elif filtered == "SHORT" and cfg.signal.sentiment_filter_enabled and sentiment_data is not None:
        if conf < cfg.signal.min_sentiment_confidence:
            filtered = "HOLD"
            reasons.append(f"low_conf={conf:.2f}<{cfg.signal.min_sentiment_confidence}")
        elif overall_sentiment == "positive":
            filtered = "HOLD"
            reasons.append("blocked:positive_sentiment(vs_SHORT)")

    # ── Risk filter (DB) ─────────────────────────────────────────────────
    risk_score = (risk_data or {}).get("composite_risk_score")

    if filtered in ("BUY", "SHORT") and cfg.signal.risk_filter_enabled and risk_score is not None:
        if risk_score > cfg.signal.max_risk_score:
            filtered = "HOLD"
            reasons.append(f"risk={risk_score:.1f}>{cfg.signal.max_risk_score}")

    return filtered, reasons
