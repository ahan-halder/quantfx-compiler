"""SQLite-backed config cache for tuned compiler configurations."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB = Path(__file__).resolve().parent / "cache" / "results.db"


def connect(db_path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS config_cache (
            source_hash TEXT NOT NULL,
            hardware_id TEXT NOT NULL,
            config_json TEXT NOT NULL,
            median_us REAL,
            p99_us REAL,
            fitness REAL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (source_hash, hardware_id, config_json)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pareto_front (
            source_hash TEXT NOT NULL,
            hardware_id TEXT NOT NULL,
            config_json TEXT NOT NULL,
            median_us REAL NOT NULL,
            p99_us REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def lookup_best(
    conn: sqlite3.Connection,
    source_hash: str,
    hardware_id: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT config_json, median_us, p99_us, fitness
        FROM config_cache
        WHERE source_hash = ? AND hardware_id = ?
        ORDER BY fitness ASC
        LIMIT 1
        """,
        (source_hash, hardware_id),
    ).fetchone()
    if row is None:
        return None
    return {
        "config": json.loads(row["config_json"]),
        "median_us": row["median_us"],
        "p99_us": row["p99_us"],
        "fitness": row["fitness"],
    }


def store_result(
    conn: sqlite3.Connection,
    *,
    source_hash: str,
    hardware_id: str,
    config: dict[str, Any],
    median_us: float,
    p99_us: float,
    fitness: float,
) -> None:
    config_json = json.dumps(config, sort_keys=True)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO config_cache
        (source_hash, hardware_id, config_json, median_us, p99_us, fitness, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (source_hash, hardware_id, config_json, median_us, p99_us, fitness, now),
    )
    conn.commit()


def store_pareto_point(
    conn: sqlite3.Connection,
    *,
    source_hash: str,
    hardware_id: str,
    config: dict[str, Any],
    median_us: float,
    p99_us: float,
) -> None:
    config_json = json.dumps(config, sort_keys=True)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO pareto_front
        (source_hash, hardware_id, config_json, median_us, p99_us, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (source_hash, hardware_id, config_json, median_us, p99_us, now),
    )
    conn.commit()


def report(conn: sqlite3.Connection, source_hash: str | None = None) -> list[dict[str, Any]]:
    query = """
        SELECT source_hash, hardware_id, config_json, median_us, p99_us, fitness, created_at
        FROM config_cache
    """
    params: tuple[Any, ...] = ()
    if source_hash:
        query += " WHERE source_hash = ?"
        params = (source_hash,)
    query += " ORDER BY fitness ASC"
    rows = conn.execute(query, params).fetchall()
    return [
        {
            "source_hash": row["source_hash"],
            "hardware_id": row["hardware_id"],
            "config": json.loads(row["config_json"]),
            "median_us": row["median_us"],
            "p99_us": row["p99_us"],
            "fitness": row["fitness"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
