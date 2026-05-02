"""Standalone backtest runner for the Trend Following strategy.

Lives at trading/trend_following/run_backtest.py — mirrors the style of
run_daily_risk.py at the trading root.

Usage (run from the trading/ root):
    trend_following\\venv\\Scripts\\python.exe trend_following\\run_backtest.py
    trend_following\\venv\\Scripts\\python.exe trend_following\\run_backtest.py --ticker AAPL
    trend_following\\venv\\Scripts\\python.exe trend_following\\run_backtest.py --start 2024-01-01
    trend_following\\venv\\Scripts\\python.exe trend_following\\run_backtest.py --json-output
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

_TRADING_ROOT = Path(__file__).resolve().parents[1]
_RISK_BACKEND = _TRADING_ROOT / "risk_calculator" / "backend"

if not _RISK_BACKEND.is_dir():
    raise SystemExit(f"risk_calculator backend not found at {_RISK_BACKEND}")

sys.path.insert(0, str(_RISK_BACKEND))
sys.path.insert(0, str(_TRADING_ROOT))

log_dir = _TRADING_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"trend_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Trend Following Backtest Runner")
    parser.add_argument("--ticker", help="Single ticker to backtest (default: all 8)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--fast", type=int, default=50, help="Fast MA window (default: 50)")
    parser.add_argument("--slow", type=int, default=200, help="Slow MA window (default: 200)")
    parser.add_argument(
        "--ma-type", dest="ma_type", choices=["SMA", "EMA"], default="EMA",
        help="Moving average type (default: EMA)",
    )
    parser.add_argument(
        "--capital", type=float, default=100_000.0,
        help="Initial capital in dollars (default: 100000)",
    )
    parser.add_argument(
        "--json-output", action="store_true",
        help="Save JSON results to trading/daily_results/trend/",
    )
    args = parser.parse_args()

    from app.services.market_data import fetch_ohlcv, fetch_risk_free_rate_annual
    from trend_following.config import TrendFollowingConfig
    from trend_following.backtest.engine import run_backtest

    cfg = TrendFollowingConfig()
    cfg.indicator.fast_period = args.fast
    cfg.indicator.slow_period = args.slow
    cfg.indicator.ma_type = args.ma_type
    cfg.backtest.initial_capital = args.capital

    if args.ticker:
        cfg.tickers = [args.ticker.upper()]

    log.info("=" * 60)
    log.info("Trend Following Backtest START")
    log.info(
        "  MA: %s(%d/%d) | Capital: $%s",
        cfg.indicator.ma_type.upper(),
        cfg.indicator.fast_period,
        cfg.indicator.slow_period,
        f"{cfg.backtest.initial_capital:,.0f}",
    )
    if args.start:
        log.info("  Start date : %s", args.start)
    if args.end:
        log.info("  End date   : %s", args.end)
    log.info("=" * 60)

    log.info("Loading price data...")
    ticker_ohlc = {}
    for ticker in cfg.tickers:
        try:
            ohlc = fetch_ohlcv(ticker, cfg.lookback_days)
            ticker_ohlc[ticker] = ohlc
            log.info("  %-6s  %d rows", ticker, len(ohlc))
        except Exception as exc:
            log.error("  %-6s  FAILED — %s", ticker, exc)

    if not ticker_ohlc:
        log.error("No price data loaded. Aborting.")
        return 1

    benchmark_names = [cfg.backtest.benchmark_ticker] + list(ticker_ohlc.keys())
    benchmark_ohlc = {}
    for b in benchmark_names:
        if b not in benchmark_ohlc:
            try:
                benchmark_ohlc[b] = fetch_ohlcv(b, cfg.lookback_days)
                log.info("  %-6s  %d rows [benchmark]", b, len(benchmark_ohlc[b]))
            except Exception as exc:
                log.warning("  %-6s  benchmark FAILED — %s", b, exc)

    try:
        rf = fetch_risk_free_rate_annual()
        log.info("Risk-free rate (annual): %.4f", rf)
    except Exception:
        rf = 0.04
        log.warning("Failed to fetch risk-free rate, using default: %.4f", rf)

    summary = run_backtest(
        cfg=cfg,
        ticker_ohlc=ticker_ohlc,
        benchmark_ohlc=benchmark_ohlc,
        rf_annual=rf,
        start_date=args.start,
        end_date=args.end,
    )

    def _fmt(v, decimals=2):
        """Format a float metric safely — returns 'N/A' for None or extreme values."""
        if v is None or abs(v) > 9999:
            return "N/A"
        return f"{v:.{decimals}f}"

    bench = cfg.backtest.benchmark_ticker.lower()
    header = (
        f"{'TICKER':<10} {'RETURN%':>8} {'CAGR%':>7} {'SHARPE':>7} {'CALMAR':>7} "
        f"{'MAX_DD%':>8} {'PF':>6} {'AVG_HOLD':>9} {'TRADES':>7} {'WIN%':>6}"
    )
    log.info("\n" + "=" * 80)
    log.info("BACKTEST RESULTS")
    log.info("=" * 80)
    log.info(header)
    log.info("-" * 85)

    def _print_row(label: str, m: dict) -> None:
        log.info(
            "%-10s %8.1f %7.1f %7s %7s %8.1f %6s %9s %7d %6.1f",
            label,
            m["total_return_pct"],
            m["cagr_pct"],
            _fmt(m.get("sharpe")),
            _fmt(m.get("calmar")),
            m["max_drawdown_pct"],
            _fmt(m.get("profit_factor")),
            f"{m.get('avg_holding_days') or 0:.0f}d",
            m["total_trades"],
            m["win_rate_pct"],
        )

    for ticker, result in summary.results.items():
        _print_row(ticker, result.metrics)

    if summary.portfolio_metrics:
        log.info("-" * 85)
        _print_row("PORTFOLIO", summary.portfolio_metrics)

    # Optionally save JSON output
    if args.json_output:
        results_dir = _TRADING_ROOT / "daily_results" / "trend"
        results_dir.mkdir(parents=True, exist_ok=True)

        output = {
            "run_at": datetime.utcnow().isoformat() + "Z",
            "config": {
                "ma_type": cfg.indicator.ma_type,
                "fast_window": cfg.indicator.fast_period,
                "slow_window": cfg.indicator.slow_period,
                "initial_capital": cfg.backtest.initial_capital,
                "benchmark": cfg.backtest.benchmark_ticker,
                "start_date": args.start,
                "end_date": args.end,
            },
        }
        for ticker, result in summary.results.items():
            output[ticker] = result.metrics
        if summary.portfolio_metrics:
            output["portfolio"] = summary.portfolio_metrics

        out_path = results_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        log.info("\nResults saved → %s", out_path)

    log.info("=" * 60)
    log.info("Trend Following Backtest END")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
