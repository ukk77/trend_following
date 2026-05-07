# Graph Report - trend_following  (2026-05-06)

## Corpus Check
- 23 files · ~10,078 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 277 nodes · 467 edges · 19 communities (13 shown, 6 thin omitted)
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 103 edges (avg confidence: 0.64)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]

## God Nodes (most connected - your core abstractions)
1. `IndicatorResult` - 24 edges
2. `_run_single_ticker()` - 18 edges
3. `Portfolio` - 18 edges
4. `ATR` - 17 edges
5. `generate_signal()` - 17 edges
6. `RSI` - 16 edges
7. `Signal` - 16 edges
8. `BacktestResult` - 15 edges
9. `BacktestSummary` - 15 edges
10. `Indicator` - 15 edges

## Surprising Connections (you probably didn't know these)
- `cmd_signals()` --calls--> `generate_signal()`  [INFERRED]
  cli.py → signals/generator.py
- `cmd_backtest()` --calls--> `run_backtest()`  [INFERRED]
  cli.py → backtest/engine.py
- `cmd_paper()` --calls--> `run_paper_trading()`  [INFERRED]
  cli.py → paper_trading/tracker.py
- `BacktestResult` --uses--> `TrendFollowingConfig`  [INFERRED]
  backtest/engine.py → config.py
- `BacktestSummary` --uses--> `TrendFollowingConfig`  [INFERRED]
  backtest/engine.py → config.py

## Communities (19 total, 6 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (17): Indicator, IndicatorResult, Standardised output container for any indicator., EMA, Moving average indicators: SMA, EMA, and CrossoverSignal.  CrossoverSignal is th, Simple Moving Average., Exponential Moving Average., SMA (+9 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (23): Volatility indicators: ATR (stop loss) and Volatility Regime filter.  ATR (Avera, Paper trading tracker., _fetch_ohlc(), Live paper trading tracker.  Processes today's trend + sentiment + risk signals, Load OHLCV data via the risk_calculator market_data service., Process today's signals and update paper positions for all tickers.      Args:, run_paper_trading(), compute_position_dollars() (+15 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (25): ADXConfig, ATRStopConfig, BacktestConfig, IndicatorConfig, MACDConfig, MultiTimeframeConfig, PositionSizingConfig, RangeFilterConfig (+17 more)

### Community 3 - "Community 3"
Cohesion: 0.1
Nodes (12): Portfolio, Simulated portfolio — tracks cash, positions, trades, and equity curve.  All pri, Execute a sell order (partial or full).          Returns:             True if sh, Liquidate the entire position for a ticker., Add one day of risk-free interest on uninvested cash., Append a daily equity snapshot., Record of a single simulated trade., Return equity curve as a pandas Series indexed by date string. (+4 more)

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (18): BacktestResult, BacktestSummary, _get_as_of(), _load_risk_history(), _load_sentiment_history(), Core backtest engine for the Trend Following strategy.  Each ticker is simulated, Per-ticker backtest output., Aggregated backtest output across all tickers. (+10 more)

### Community 5 - "Community 5"
Cohesion: 0.12
Nodes (9): MACD, Momentum indicators: RSI and MACD.  RSI  — Relative Strength Index (Wilder smoot, Return True if MACD histogram is positive (bullish)., Return True if MACD histogram is negative (bearish)., Relative Strength Index using Wilder's smoothing (EWMA with alpha=1/period)., Return True if the latest RSI is overbought., Return True if the latest RSI is oversold., Moving Average Convergence Divergence.      Components:         macd_line    = E (+1 more)

### Community 6 - "Community 6"
Cohesion: 0.13
Nodes (10): ABC, compute(), Indicator, Abstract base class for all trend indicators.  New indicators can be added by su, Common interface for all trend indicators.      Subclasses must implement `compu, Return a normalised signal series in [-1.0, +1.0].          Default implementati, ADX, Trend strength indicator: ADX (Average Directional Index).  ADX measures trend s (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.16
Nodes (18): alpha_vs_benchmark(), cagr(), compute_all_metrics(), _log_returns(), max_drawdown(), Performance metrics for backtest results.  All functions accept a pandas Series, Compute and return all performance metrics as a flat dictionary.      Args:, Total percentage return (e.g. 0.35 = +35%). (+10 more)

### Community 8 - "Community 8"
Cohesion: 0.13
Nodes (8): ATR, Return size multiplier [min_mult .. 1.0] based on volatility regime., Return the current position-size multiplier based on volatility., Average True Range using Wilder smoothing.      Returns ATR values as the `value, Compute ATR-based stop price.          Args:             ohlc: OHLCV DataFrame., Return the full ATR series for use in backtest simulation., Realised volatility regime filter.      Computes 30-day annualised realised vola, VolatilityRegime

### Community 9 - "Community 9"
Cohesion: 0.19
Nodes (18): get_cash_balance(), _get_conn(), get_portfolio_snapshot(), get_positions(), get_trades(), init_db(), log_trade(), SQLite storage for paper trading positions and trade history.  DB file: trend_fo (+10 more)

### Community 10 - "Community 10"
Cohesion: 0.15
Nodes (13): cmd_backtest(), cmd_paper(), cmd_positions(), cmd_signals(), Command-line interface for the Trend Following strategy.  Usage (run from the tr, Run paper trading — process today's signals and update positions., Show current open paper positions with mark-to-market P&L., Print current BUY/SELL/HOLD signals for all (or one) ticker. (+5 more)

### Community 11 - "Community 11"
Cohesion: 0.21
Nodes (5): Volume confirmation indicator.  Confirms trend signals only when volume is above, Volume-based signal confirmation filter.      signal_series() returns:         +, Return True if latest volume confirms the signal., Return current volume / MA(volume) ratio., VolumeConfirmation

## Knowledge Gaps
- **112 isolated node(s):** `Command-line interface for the Trend Following strategy.  Usage (run from the tr`, `Print current BUY/SELL/HOLD signals for all (or one) ticker.`, `Run a full historical backtest and print results.`, `Run paper trading — process today's signals and update positions.`, `Show current open paper positions with mark-to-market P&L.` (+107 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_run_single_ticker()` connect `Community 4` to `Community 0`, `Community 1`, `Community 3`, `Community 5`, `Community 6`, `Community 7`, `Community 8`, `Community 11`?**
  _High betweenness centrality (0.145) - this node is a cross-community bridge._
- **Why does `Portfolio` connect `Community 3` to `Community 4`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Why does `BacktestSummary` connect `Community 4` to `Community 0`, `Community 1`, `Community 3`, `Community 5`, `Community 6`, `Community 8`, `Community 10`, `Community 11`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Are the 22 inferred relationships involving `IndicatorResult` (e.g. with `RSI` and `MACD`) actually correct?**
  _`IndicatorResult` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `_run_single_ticker()` (e.g. with `CrossoverSignal` and `ADX`) actually correct?**
  _`_run_single_ticker()` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Portfolio` (e.g. with `BacktestResult` and `BacktestSummary`) actually correct?**
  _`Portfolio` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `ATR` (e.g. with `BacktestResult` and `BacktestSummary`) actually correct?**
  _`ATR` has 8 INFERRED edges - model-reasoned connections that need verification._