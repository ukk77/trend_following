---
description: Regenerate graphify code graphs for all 4 trading services and merge into a cross-service graph
---

1. Run each update command sequentially from `c:\Users\ukard\OneDrive\Desktop\trading`. Wait for each to complete before starting the next:
   - `python -m graphify update sentiment_analysis`
   - `python -m graphify update risk_calculator`
   - `python -m graphify update mean_reversion`
   - `python -m graphify update trend_following`

2. Merge all four service graphs into the top-level cross-service graph:
   `python -m graphify merge-graphs sentiment_analysis\graphify-out\graph.json risk_calculator\graphify-out\graph.json mean_reversion\graphify-out\graph.json trend_following\graphify-out\graph.json --output graphify-out`

3. Confirm the merged graph was written to `graphify-out\services-graph.html` and report the node/edge counts from each updated service's `GRAPH_REPORT.md`.
