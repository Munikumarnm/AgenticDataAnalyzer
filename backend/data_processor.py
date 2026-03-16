"""
data_processor.py — Data cleaning and anomaly-detection pipeline for DataCopilot.

Pipeline steps (applied in order):
  1. Normalize column names  (lowercase, underscores, strip punctuation)
  2. Fill missing values      (median for numeric, "Unknown" for text)
  3. Parse date columns       (auto-detect by column name hints)
  4. Remove anomalies         (optional — IQR method, per numeric column)
"""

import pandas as pd


# ── Step 1: Column name normalization ─────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        for c in df.columns
    ]
    return df


# ── Step 2: Fill missing values ────────────────────────────────────────────────

def _fill_missing(df: pd.DataFrame) -> tuple:
    """
    Returns (cleaned_df, filled_dict) where filled_dict maps
    column_name → number_of_cells_that_were_filled.
    """
    df = df.copy()
    filled: dict = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            median = df[col].median(skipna=True)
            df[col] = df[col].fillna(0.0 if pd.isna(median) else median)
        else:
            df[col] = df[col].fillna("Unknown")
        filled[col] = n_missing
    return df, filled


# ── Step 3: Date column parsing ────────────────────────────────────────────────

def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_hints = {"date", "joined", "time", "created", "updated", "birth", "dob", "timestamp"}
    for col in df.columns:
        if any(hint in col.lower() for hint in date_hints):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


# ── Step 4: Anomaly detection (IQR method) ────────────────────────────────────

def _anomaly_mask(df: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean Series: True where a row is an outlier in at least
    one numeric column.  Uses IQR × 1.5 fence (Tukey's method).
    Columns with fewer than 10 non-null values or zero IQR are skipped.
    """
    mask = pd.Series(False, index=df.index)
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outliers = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        mask |= outliers.fillna(False)
    return mask


# ── Public API ─────────────────────────────────────────────────────────────────

def process_upload(df: pd.DataFrame, remove_anomalies: bool = False) -> tuple:
    """
    Full data-processing pipeline.

    Parameters
    ----------
    df               : raw DataFrame from file upload
    remove_anomalies : if True, IQR outlier rows are dropped

    Returns
    -------
    (clean_df, report)  where report is a dict with processing stats
    """
    rows_original = len(df)

    df = _normalize_columns(df)
    df, filled = _fill_missing(df)
    df = _parse_dates(df)

    anomalies_removed = 0
    if remove_anomalies:
        mask = _anomaly_mask(df)
        anomalies_removed = int(mask.sum())
        df = df[~mask].reset_index(drop=True)

    report = {
        "rows_original": rows_original,
        "rows_cleaned": len(df),
        "missing_filled": filled,          # {col: count}
        "anomalies_removed": anomalies_removed,
    }
    return df, report
