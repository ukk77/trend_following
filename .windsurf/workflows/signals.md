---
description: Generate BUY/SELL/HOLD signals for the Trend Following strategy
---

1. Ask the user if they want signals for all configured tickers or a specific ticker (e.g. AAPL).

2. Run from the trading root `c:\Users\ukard\OneDrive\Desktop\trading`:
   - All tickers: `trend_following\venv\Scripts\python.exe -m trend_following.cli signals`
   - Single ticker: `trend_following\venv\Scripts\python.exe -m trend_following.cli signals --ticker <TICKER>`
   - JSON output: append `--json` to either command above.

3. The output columns are: TICKER, ACTION (BUY/SELL/HOLD), DIR (trend direction), STRENGTH, SENTIMENT, RISK_SCORE, REASON.

4. Summarize any BUY or SELL signals — note the trend direction (+1 = uptrend, -1 = downtrend), filtered strength, and risk bucket.
