"""SQLite storage for paper trading positions and trade history.

DB file: trend_following/paper_trades.db  (auto-created on first use)

Tables:
    paper_positions — one row per open ticker (upserted on each trade)
    paper_trades    — append-only trade log
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).resolve().parents[1] / "paper_trades.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables and indexes if they do not exist."""
    with _get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS paper_positions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL UNIQUE,
                shares      INTEGER NOT NULL,
                avg_cost    REAL    NOT NULL,
                opened_at   TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS paper_trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                action          TEXT    NOT NULL,
                shares          INTEGER NOT NULL,
                price           REAL    NOT NULL,
                commission      REAL    NOT NULL DEFAULT 0,
                pnl             REAL,
                executed_at     TEXT    NOT NULL,
                reason          TEXT,
                sentiment       TEXT,
                risk_score      REAL,
                signal_strength REAL
            );

            CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker
                ON paper_trades(ticker, executed_at);

            CREATE TABLE IF NOT EXISTS daily_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date    TEXT    NOT NULL UNIQUE,
                run_at      TEXT    NOT NULL,
                tickers_processed INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.commit()
        try:
            conn.execute("ALTER TABLE paper_positions ADD COLUMN atr_stop REAL")
            conn.commit()
        except Exception:
            pass  # column already exists
        try:
            conn.execute("ALTER TABLE paper_positions ADD COLUMN peak_price REAL")
            conn.commit()
        except Exception:
            pass  # column already exists


def upsert_position(ticker: str, shares: int, avg_cost: float, atr_stop: Optional[float] = None) -> None:
    """Insert or update an open position. Removes the row when shares == 0."""
    init_db()
    now = datetime.utcnow().isoformat() + "Z"
    with _get_conn() as conn:
        if shares > 0:
            conn.execute(
                """
                INSERT INTO paper_positions (ticker, shares, avg_cost, atr_stop, opened_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    shares     = excluded.shares,
                    avg_cost   = excluded.avg_cost,
                    atr_stop   = excluded.atr_stop,
                    updated_at = excluded.updated_at
                """,
                (ticker.upper(), shares, avg_cost, atr_stop, now, now),
            )
        else:
            conn.execute(
                "DELETE FROM paper_positions WHERE UPPER(ticker)=UPPER(?)",
                (ticker.upper(),),
            )
        conn.commit()


def log_trade(
    ticker: str,
    action: str,
    shares: int,
    price: float,
    commission: float = 0.0,
    pnl: Optional[float] = None,
    reason: str = "",
    sentiment: Optional[str] = None,
    risk_score: Optional[float] = None,
    signal_strength: Optional[float] = None,
) -> None:
    """Append one trade record to the trade log."""
    init_db()
    now = datetime.utcnow().isoformat() + "Z"
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO paper_trades
                (ticker, action, shares, price, commission, pnl,
                 executed_at, reason, sentiment, risk_score, signal_strength)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker.upper(), action, shares, price, commission, pnl,
                now, reason, sentiment, risk_score, signal_strength,
            ),
        )
        conn.commit()


def update_atr_stop(ticker: str, new_stop: float) -> None:
    """Update the stored ATR trailing stop for an open position."""
    init_db()
    now = datetime.utcnow().isoformat() + "Z"
    with _get_conn() as conn:
        conn.execute(
            "UPDATE paper_positions SET atr_stop=?, updated_at=? WHERE UPPER(ticker)=UPPER(?)",
            (new_stop, now, ticker.upper()),
        )
        conn.commit()


def update_peak_price(ticker: str, peak: float) -> None:
    """Update the stored peak price for an open position."""
    init_db()
    now = datetime.utcnow().isoformat() + "Z"
    with _get_conn() as conn:
        conn.execute(
            "UPDATE paper_positions SET peak_price=?, updated_at=? WHERE UPPER(ticker)=UPPER(?)",
            (peak, now, ticker.upper()),
        )
        conn.commit()


def get_cash_balance(initial_capital: float) -> float:
    """Compute current cash from initial capital minus buys plus sell proceeds."""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT action, shares, price, commission FROM paper_trades"
        ).fetchall()
    cash = initial_capital
    for row in rows:
        if row["action"] == "BUY":
            cash -= row["shares"] * row["price"] + row["commission"]
        elif row["action"] == "SELL":
            cash += row["shares"] * row["price"] - row["commission"]
    return cash


def get_positions() -> List[Dict[str, Any]]:
    """Return all open paper positions."""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM paper_positions ORDER BY ticker"
        ).fetchall()
    return [dict(r) for r in rows]


def get_trades(
    ticker: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Return recent paper trades, optionally filtered by ticker."""
    init_db()
    params: List[Any] = []
    where = ""
    if ticker:
        where = "WHERE UPPER(ticker)=UPPER(?)"
        params.append(ticker.upper())
    with _get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM paper_trades {where} ORDER BY executed_at DESC LIMIT ?",
            [*params, limit],
        ).fetchall()
    return [dict(r) for r in rows]


def get_portfolio_snapshot(current_prices: Dict[str, float]) -> Dict[str, Any]:
    """Return mark-to-market snapshot of all open positions.

    Args:
        current_prices: Dict of ticker -> current price.

    Returns:
        Dict with positions list and aggregate P&L totals.
    """
    positions = get_positions()
    total_market_value = 0.0
    total_cost_basis = 0.0
    pos_details = []

    for pos in positions:
        ticker = pos["ticker"]
        shares = pos["shares"]
        avg_cost = pos["avg_cost"]
        price = current_prices.get(ticker, avg_cost)
        market_value = shares * price
        cost_value = shares * avg_cost
        unrealised_pnl = market_value - cost_value

        total_market_value += market_value
        total_cost_basis += cost_value
        pos_details.append(
            {
                "ticker": ticker,
                "shares": shares,
                "avg_cost": round(avg_cost, 4),
                "current_price": round(price, 4),
                "market_value": round(market_value, 2),
                "unrealised_pnl": round(unrealised_pnl, 2),
                "unrealised_pnl_pct": round(
                    unrealised_pnl / cost_value * 100 if cost_value > 0 else 0.0, 2
                ),
            }
        )

    return {
        "positions": pos_details,
        "total_market_value": round(total_market_value, 2),
        "total_cost_basis": round(total_cost_basis, 2),
        "total_unrealised_pnl": round(total_market_value - total_cost_basis, 2),
        "total_unrealised_pnl_pct": round(
            (total_market_value / total_cost_basis - 1.0) * 100
            if total_cost_basis > 0
            else 0.0,
            2,
        ),
    }


def has_run_today() -> bool:
    """Return True if paper trading has already been recorded for today's date."""
    today = datetime.utcnow().date().isoformat()
    with _get_conn() as conn:
        try:
            row = conn.execute(
                "SELECT 1 FROM daily_runs WHERE run_date = ?", (today,)
            ).fetchone()
        except sqlite3.OperationalError:
            return False
    return row is not None


def record_daily_run(tickers_processed: int) -> None:
    """Insert or replace today's run record in daily_runs."""
    today = datetime.utcnow().date().isoformat()
    now = datetime.utcnow().isoformat() + "Z"
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO daily_runs (run_date, run_at, tickers_processed)
            VALUES (?, ?, ?)
            ON CONFLICT(run_date) DO UPDATE SET run_at=excluded.run_at,
                tickers_processed=excluded.tickers_processed
            """,
            (today, now, tickers_processed),
        )
        conn.commit()
