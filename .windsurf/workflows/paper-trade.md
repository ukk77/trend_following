---
description: Process today's paper trading signals and show positions for Trend Following
---

1. Process today's signals and update paper positions. Run from `c:\Users\ukard\OneDrive\Desktop\trading`:
   `trend_following\venv\Scripts\python.exe -m trend_following.cli paper`
   This evaluates current MA crossover signals against open positions and executes BUY/SELL/HOLD actions in the paper portfolio (`paper_trades.db`).

2. Show current open positions with mark-to-market P&L:
   `trend_following\venv\Scripts\python.exe -m trend_following.cli positions`

3. Summarize:
   - Actions taken today (BUY/SELL with ticker, shares, price)
   - Any realised P&L from closed trades
   - Total portfolio unrealised P&L and position count
