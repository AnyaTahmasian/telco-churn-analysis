from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from .data import PROJECT_ROOT

SQL_FILE = PROJECT_ROOT / "sql" / "eda_queries.sql"


def build_database(df: pd.DataFrame, path: str | Path = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    df.to_sql("customers", conn, index=False, if_exists="replace")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_contract ON customers (Contract)")
    return conn


def load_named_queries(path: str | Path = SQL_FILE) -> dict[str, str]:
    # each query in the file starts with a "-- name: <key>" line
    text = Path(path).read_text(encoding="utf-8")
    parts = re.split(r"^--\s*name:\s*(\w+)\s*$", text, flags=re.MULTILINE)
    return {name: body.strip() for name, body in zip(parts[1::2], parts[2::2])}


def run_query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)
