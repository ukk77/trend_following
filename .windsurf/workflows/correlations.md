---
description: Compute long-term return correlations for all MR + TF tickers and report concentration risk
---

Run the cross-strategy correlation analysis from the trading root. Fetches 3 years of daily prices via yfinance and computes Pearson correlations of log returns for all 22 tickers across both strategies (including SPY/QQQ benchmarks, IWM and GLD diversifiers).

1. Run the standard 3-year correlation report:
   `trend_following\venv\Scripts\python.exe compute_correlations.py`
   Run from `c:\Users\ukard\OneDrive\Desktop\trading`.

2. For a 1-year short-term view (useful after market regime changes):
   `trend_following\venv\Scripts\python.exe compute_correlations.py --years 1`
   Run from `c:\Users\ukard\OneDrive\Desktop\trading`.

3. For a 5-year long-term view:
   `trend_following\venv\Scripts\python.exe compute_correlations.py --years 5`
   Run from `c:\Users\ukard\OneDrive\Desktop\trading`.

4. Review the output sections:
   - **Heatmap** — visual overview of all pairwise correlations
   - **Highly Correlated Pairs** — pairs with r ≥ 0.70; flag any currently held together
   - **Diversifying Pairs** — pairs with r ≤ 0.35; safe to hold simultaneously
   - **Clusters** — auto-grouped tickers that move together (reduce combined position sizes within a cluster)
   - **Portfolio Concentration** — correlations between your current open positions only
   - **Beta to SPY** — how much each ticker amplifies or dampens market moves

5. Act on findings:
   - Two held positions with r ≥ 0.75 → consider halving one position
   - GLD or IWM in portfolio → confirms diversification is active
   - Beta > 1.3 on a held position → verify risk score allows it
