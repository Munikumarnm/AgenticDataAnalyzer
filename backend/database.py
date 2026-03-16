"""
database.py — SQLite persistence layer for DataCopilot.
All data operations (save, query, schema inspection) go through here.
"""
import os
import sqlite3

import pandas as pd

# Place the database in the project root directory
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_store.db"
)

TABLE = "uploaded_data"


def save_dataframe(df: pd.DataFrame) -> None:
    """Persist a DataFrame to SQLite, replacing any existing table."""
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE, conn, if_exists="replace", index=False)
    conn.close()


def get_schema() -> tuple:
    """
    Return (all_columns, numeric_columns) based on the current table schema.
    Numeric columns are those stored as INT/REAL/FLOAT/NUMERIC/DOUBLE/DECIMAL.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({TABLE})")
    rows = cursor.fetchall()
    conn.close()

    numeric_types = {"INT", "INTEGER", "REAL", "FLOAT", "NUMERIC", "DOUBLE", "DECIMAL"}
    columns = [r[1] for r in rows]
    numeric_cols = [r[1] for r in rows if str(r[2]).upper() in numeric_types]
    return columns, numeric_cols


def run_query(sql: str) -> pd.DataFrame:
    """Execute a SELECT statement and return results as a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


def read_all() -> pd.DataFrame:
    """Read the entire uploaded_data table."""
    return run_query(f"SELECT * FROM {TABLE}")


def has_data() -> bool:
    """Return True if the uploaded_data table exists and contains at least one row."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)
        )
        exists = cursor.fetchone() is not None
        conn.close()
        if not exists:
            return False
        result = run_query(f"SELECT COUNT(*) AS cnt FROM {TABLE}")
        return int(result["cnt"].iloc[0]) > 0
    except Exception:
        return False
