# trading/database.py
"""
SQLite persistence layer for the Solana paper-trading bot.

Tables
------
wallets              — tracked wallets with their profit statistics
wallet_transactions  — raw swap transactions fetched from the chain
patterns             — detected trading patterns and their success rates
paper_trades         — every paper trade executed by the bot
portfolio            — current paper-portfolio snapshot (one row per token)
sentiment_data       — sentiment records from news/social sources
manipulation_flags   — detected market manipulation flags per token
technical_indicators — computed technical indicator values per token
"""

import sqlite3
import logging
import os
from typing import Optional

import config

logger = logging.getLogger(__name__)


def _connect() -> sqlite3.Connection:
    """Return a thread-local SQLite connection with row_factory set."""
    db_path = os.path.abspath(config.TRADING_DB_PATH)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> sqlite3.Connection:
    """Create all tables if they do not exist and return the connection."""
    conn = _connect()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS wallets (
            address         TEXT PRIMARY KEY,
            label           TEXT,
            total_trades    INTEGER DEFAULT 0,
            winning_trades  INTEGER DEFAULT 0,
            total_pnl_usd   REAL DEFAULT 0.0,
            win_rate        REAL DEFAULT 0.0,
            first_seen      TEXT,
            last_updated    TEXT,
            is_tracked      INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS wallet_transactions (
            signature       TEXT PRIMARY KEY,
            wallet_address  TEXT NOT NULL,
            block_time      INTEGER,
            token_in        TEXT,
            token_out       TEXT,
            amount_in       REAL,
            amount_out      REAL,
            price_usd       REAL,
            dex_program     TEXT,
            is_profitable   INTEGER,
            fetched_at      TEXT,
            FOREIGN KEY (wallet_address) REFERENCES wallets(address)
        );

        CREATE TABLE IF NOT EXISTS patterns (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL,
            description     TEXT,
            token_address   TEXT,
            dex_program     TEXT,
            min_hold_secs   INTEGER,
            max_hold_secs   INTEGER,
            avg_entry_usd   REAL,
            avg_roi_pct     REAL,
            sample_size     INTEGER DEFAULT 0,
            success_rate    REAL DEFAULT 0.0,
            last_seen       TEXT,
            created_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS paper_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_id      INTEGER,
            token_address   TEXT NOT NULL,
            token_symbol    TEXT,
            action          TEXT NOT NULL,   -- 'BUY' or 'SELL'
            virtual_amount  REAL NOT NULL,   -- in USD
            token_qty       REAL,
            price_usd       REAL,
            pnl_usd         REAL,
            pnl_pct         REAL,
            outcome         TEXT,            -- 'WIN' | 'LOSS' | 'OPEN'
            signal_score    REAL,
            executed_at     TEXT,
            closed_at       TEXT,
            FOREIGN KEY (pattern_id) REFERENCES patterns(id)
        );

        CREATE TABLE IF NOT EXISTS portfolio (
            token_address   TEXT PRIMARY KEY,
            token_symbol    TEXT,
            qty             REAL DEFAULT 0.0,
            avg_cost_usd    REAL DEFAULT 0.0,
            current_price   REAL DEFAULT 0.0,
            unrealised_pnl  REAL DEFAULT 0.0,
            last_updated    TEXT
        );

        CREATE TABLE IF NOT EXISTS bot_state (
            key     TEXT PRIMARY KEY,
            value   TEXT
        );

        CREATE TABLE IF NOT EXISTS sentiment_data (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            token_address   TEXT,
            source          TEXT NOT NULL,     -- 'news', 'twitter', 'reddit', 'discord'
            headline        TEXT,
            sentiment_score REAL DEFAULT 0.0,  -- -1.0 (bearish) to +1.0 (bullish)
            relevance_score REAL DEFAULT 0.0,  -- 0.0 to 1.0
            raw_text        TEXT,
            fetched_at      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS manipulation_flags (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            token_address   TEXT NOT NULL,
            flag_type       TEXT NOT NULL,     -- 'WASH_TRADE', 'PUMP_SCHEME', 'HONEYPOT', 'FAKE_VOLUME', 'COORDINATED'
            severity        REAL DEFAULT 0.0,  -- 0.0 to 1.0
            evidence        TEXT,              -- JSON description of evidence
            wallet_addresses TEXT,             -- comma-separated involved wallets
            detected_at     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS technical_indicators (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            token_address   TEXT NOT NULL,
            indicator_name  TEXT NOT NULL,     -- 'RSI', 'MACD', 'BBANDS', 'VOLUME_SMA'
            value           REAL,
            signal_value    REAL,             -- e.g. MACD signal line
            upper_band      REAL,             -- e.g. Bollinger upper
            lower_band      REAL,             -- e.g. Bollinger lower
            timeframe       TEXT DEFAULT '1h',
            computed_at     TEXT NOT NULL
        );
    """)
    conn.commit()

    # Seed initial virtual cash balance if not present
    cursor.execute(
        "INSERT OR IGNORE INTO bot_state (key, value) VALUES ('virtual_cash_usd', ?)",
        (str(config.PAPER_TRADING_STARTING_BALANCE),),
    )
    conn.commit()
    logger.info("Database initialised at %s", config.TRADING_DB_PATH)
    return conn


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_virtual_cash(conn: sqlite3.Connection) -> float:
    row = conn.execute(
        "SELECT value FROM bot_state WHERE key='virtual_cash_usd'"
    ).fetchone()
    return float(row["value"]) if row else config.PAPER_TRADING_STARTING_BALANCE


def set_virtual_cash(conn: sqlite3.Connection, amount: float) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO bot_state (key, value) VALUES ('virtual_cash_usd', ?)",
        (str(amount),),
    )
    conn.commit()


def upsert_wallet(conn: sqlite3.Connection, address: str, label: Optional[str] = None) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO wallets (address, label, first_seen, last_updated)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(address) DO UPDATE SET
            label        = COALESCE(excluded.label, label),
            last_updated = excluded.last_updated
        """,
        (address, label, now, now),
    )
    conn.commit()


def update_wallet_stats(
    conn: sqlite3.Connection,
    address: str,
    total_trades: int,
    winning_trades: int,
    total_pnl_usd: float,
) -> None:
    from datetime import datetime, timezone
    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
    conn.execute(
        """
        UPDATE wallets
        SET total_trades   = ?,
            winning_trades = ?,
            total_pnl_usd  = ?,
            win_rate       = ?,
            last_updated   = ?
        WHERE address = ?
        """,
        (total_trades, winning_trades, total_pnl_usd, win_rate,
         __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
         address),
    )
    conn.commit()


def insert_transaction(conn: sqlite3.Connection, tx: dict) -> None:
    from datetime import datetime, timezone
    conn.execute(
        """
        INSERT OR IGNORE INTO wallet_transactions
            (signature, wallet_address, block_time, token_in, token_out,
             amount_in, amount_out, price_usd, dex_program, is_profitable, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            tx.get("signature"),
            tx.get("wallet_address"),
            tx.get("block_time"),
            tx.get("token_in"),
            tx.get("token_out"),
            tx.get("amount_in"),
            tx.get("amount_out"),
            tx.get("price_usd"),
            tx.get("dex_program"),
            tx.get("is_profitable"),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def insert_pattern(conn: sqlite3.Connection, pattern: dict) -> int:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO patterns
            (name, description, token_address, dex_program,
             min_hold_secs, max_hold_secs, avg_entry_usd,
             avg_roi_pct, sample_size, success_rate, last_seen, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pattern.get("name"),
            pattern.get("description"),
            pattern.get("token_address"),
            pattern.get("dex_program"),
            pattern.get("min_hold_secs"),
            pattern.get("max_hold_secs"),
            pattern.get("avg_entry_usd"),
            pattern.get("avg_roi_pct"),
            pattern.get("sample_size", 0),
            pattern.get("success_rate", 0.0),
            pattern.get("last_seen", now),
            now,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def record_paper_trade(conn: sqlite3.Connection, trade: dict) -> int:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO paper_trades
            (pattern_id, token_address, token_symbol, action,
             virtual_amount, token_qty, price_usd, pnl_usd, pnl_pct,
             outcome, signal_score, executed_at, closed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trade.get("pattern_id"),
            trade.get("token_address"),
            trade.get("token_symbol"),
            trade.get("action"),
            trade.get("virtual_amount"),
            trade.get("token_qty"),
            trade.get("price_usd"),
            trade.get("pnl_usd"),
            trade.get("pnl_pct"),
            trade.get("outcome", "OPEN"),
            trade.get("signal_score"),
            trade.get("executed_at", now),
            trade.get("closed_at"),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_open_paper_trades(conn: sqlite3.Connection) -> list:
    rows = conn.execute(
        "SELECT * FROM paper_trades WHERE outcome = 'OPEN' ORDER BY executed_at ASC"
    ).fetchall()
    return [dict(r) for r in rows]


def close_paper_trade(
    conn: sqlite3.Connection,
    trade_id: int,
    pnl_usd: float,
    pnl_pct: float,
    outcome: str,
) -> None:
    from datetime import datetime, timezone
    conn.execute(
        """
        UPDATE paper_trades
        SET pnl_usd   = ?,
            pnl_pct   = ?,
            outcome   = ?,
            closed_at = ?
        WHERE id = ?
        """,
        (pnl_usd, pnl_pct, outcome,
         __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
         trade_id),
    )
    conn.commit()


def upsert_portfolio_position(conn: sqlite3.Connection, position: dict) -> None:
    from datetime import datetime, timezone
    conn.execute(
        """
        INSERT INTO portfolio
            (token_address, token_symbol, qty, avg_cost_usd,
             current_price, unrealised_pnl, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(token_address) DO UPDATE SET
            token_symbol   = excluded.token_symbol,
            qty            = excluded.qty,
            avg_cost_usd   = excluded.avg_cost_usd,
            current_price  = excluded.current_price,
            unrealised_pnl = excluded.unrealised_pnl,
            last_updated   = excluded.last_updated
        """,
        (
            position["token_address"],
            position.get("token_symbol"),
            position.get("qty", 0.0),
            position.get("avg_cost_usd", 0.0),
            position.get("current_price", 0.0),
            position.get("unrealised_pnl", 0.0),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def get_performance_summary(conn: sqlite3.Connection) -> dict:
    """Return high-level stats about the paper-trading session."""
    row = conn.execute(
        """
        SELECT
            COUNT(*)                                    AS total_trades,
            SUM(CASE WHEN outcome='WIN'  THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) AS losses,
            SUM(CASE WHEN outcome='OPEN' THEN 1 ELSE 0 END) AS open_trades,
            COALESCE(SUM(pnl_usd), 0.0)                AS total_pnl_usd,
            COALESCE(AVG(CASE WHEN outcome != 'OPEN' THEN pnl_pct END), 0.0) AS avg_roi_pct
        FROM paper_trades
        """
    ).fetchone()
    summary = dict(row) if row else {}
    closed = (summary.get("wins", 0) or 0) + (summary.get("losses", 0) or 0)
    summary["win_rate_pct"] = (
        (summary["wins"] / closed * 100.0) if closed > 0 else 0.0
    )
    summary["virtual_cash_usd"] = get_virtual_cash(conn)
    return summary


def insert_sentiment(conn: sqlite3.Connection, data: dict) -> int:
    """Insert a sentiment data record and return its row id."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO sentiment_data
            (token_address, source, headline, sentiment_score,
             relevance_score, raw_text, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data.get("token_address"),
            data.get("source", "unknown"),
            data.get("headline"),
            data.get("sentiment_score", 0.0),
            data.get("relevance_score", 0.0),
            data.get("raw_text"),
            data.get("fetched_at", now),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_recent_sentiment(
    conn: sqlite3.Connection,
    token_address: str,
    limit: int = 50,
) -> list[dict]:
    """Return recent sentiment records for a token."""
    rows = conn.execute(
        """
        SELECT * FROM sentiment_data
        WHERE token_address = ?
        ORDER BY fetched_at DESC
        LIMIT ?
        """,
        (token_address, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_aggregate_sentiment(conn: sqlite3.Connection, token_address: str) -> float:
    """Return the average sentiment score for a token (recent 50 entries)."""
    row = conn.execute(
        """
        SELECT AVG(sentiment_score) AS avg_sentiment
        FROM (
            SELECT sentiment_score FROM sentiment_data
            WHERE token_address = ?
            ORDER BY fetched_at DESC
            LIMIT 50
        )
        """,
        (token_address,),
    ).fetchone()
    return float(row["avg_sentiment"]) if row and row["avg_sentiment"] is not None else 0.0


def insert_manipulation_flag(conn: sqlite3.Connection, flag: dict) -> int:
    """Insert a manipulation flag and return its row id."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO manipulation_flags
            (token_address, flag_type, severity, evidence,
             wallet_addresses, detected_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            flag.get("token_address"),
            flag.get("flag_type"),
            flag.get("severity", 0.0),
            flag.get("evidence"),
            flag.get("wallet_addresses"),
            flag.get("detected_at", now),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_manipulation_flags(
    conn: sqlite3.Connection,
    token_address: str,
    hours: int = 24,
) -> list[dict]:
    """Return manipulation flags for a token from the last *hours*."""
    interval = f"-{hours} hours"
    rows = conn.execute(
        """
        SELECT * FROM manipulation_flags
        WHERE token_address = ?
          AND detected_at >= datetime('now', ?)
        ORDER BY detected_at DESC
        """,
        (token_address, interval),
    ).fetchall()
    return [dict(r) for r in rows]


def is_token_flagged(conn: sqlite3.Connection, token_address: str, min_severity: float = 0.5) -> bool:
    """Check if a token has recent high-severity manipulation flags."""
    row = conn.execute(
        """
        SELECT COUNT(*) AS cnt FROM manipulation_flags
        WHERE token_address = ?
          AND severity >= ?
          AND detected_at >= datetime('now', '-24 hours')
        """,
        (token_address, min_severity),
    ).fetchone()
    return (row["cnt"] or 0) > 0


def insert_technical_indicator(conn: sqlite3.Connection, indicator: dict) -> int:
    """Insert a technical indicator record and return its row id."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO technical_indicators
            (token_address, indicator_name, value, signal_value,
             upper_band, lower_band, timeframe, computed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            indicator.get("token_address"),
            indicator.get("indicator_name"),
            indicator.get("value"),
            indicator.get("signal_value"),
            indicator.get("upper_band"),
            indicator.get("lower_band"),
            indicator.get("timeframe", "1h"),
            indicator.get("computed_at", now),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_latest_indicator(
    conn: sqlite3.Connection,
    token_address: str,
    indicator_name: str,
) -> Optional[dict]:
    """Return the most recent value of a specific indicator for a token."""
    row = conn.execute(
        """
        SELECT * FROM technical_indicators
        WHERE token_address = ? AND indicator_name = ?
        ORDER BY computed_at DESC
        LIMIT 1
        """,
        (token_address, indicator_name),
    ).fetchone()
    return dict(row) if row else None
