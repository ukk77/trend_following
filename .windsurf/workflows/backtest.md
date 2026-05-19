---
description: Run a historical backtest for the Trend Following strategy
---

1. Ask the user for any of the following optional parameters (skip any not provided):
   - `--ticker TICKER` — single symbol, or all configured tickers by default
   - `--start YYYY-MM-DD` — backtest start date
   - `--end YYYY-MM-DD` — backtest end date (default: today)
   - `--fast N` — fast MA window in days (default: 20)
   - `--slow N` — slow MA window in days (default: 50)
   - `--ma-type SMA|EMA` — moving average type (default: EMA)
   - `--capital N` — initial capital in USD (default: 100000)

2. Build and run the command from `c:\Users\ukard\OneDrive\Desktop\trading`:
   `trend_following\venv\Scripts\python.exe -m trend_following.cli backtest [--ticker TICKER] [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--fast N] [--slow N] [--ma-type SMA|EMA] [--capital N]`

3. The results table shows per-ticker: RETURN%, CAGR%, SHARPE, MAX_DD%, ALPHA_SPY%, TRADES, WIN%.

4. Summarize key findings — flag any tickers with Sharpe < 0.5, max drawdown > 20%, or negative alpha vs benchmark as needing review.
