# preprocess_merged_sentiment.py
"""
Robust preprocessing for merged sentiment + Nifty OHLC data.

Saves: preprocessed_nifty_sentiment.csv

Run:
    python preprocess_merged_sentiment.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib

INPUT_CSV = "merged_sentiment_nifty.csv"   # change if needed
OUTPUT_CSV = "preprocessed_nifty_sentiment.csv"

# Common candidate/trading date column names (priority order)
DATE_CANDIDATES = ["candidate", "trading_date", "tradingday", "tradingday_date", "orig_date", "date"]

# Common variants for OHLC and volume columns — we'll choose the first match we find
OHLC_VARIANTS = {
    "open": ["open", "open_0", "open_first", "open_x", "o"],
    "high": ["high", "high_0", "high_first", "high_x", "h"],
    "low":  ["low", "low_0", "low_first", "low_x", "l"],
    "close":["close", "close_0", "close_first", "close_x", "adj_close", "c"],
    "volume":["volume", "vol", "volume_0", "volume_first", "volume_x"]
}
SENT_SCORE_VARIANTS = ["sentiment_score", "sent_score", "score"]
SENT_LABEL_VARIANTS = ["sentiment_label", "sent_label", "label"]

def read_df(path):
    p = Path(path)
    if not p.exists():
        print(f"Input file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(p)
    # normalize whitespace in column names
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_column(df, candidates):
    """Return the first column name from candidates that exists in df.columns (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}  # map lowercase -> original name
    for cand in candidates:
        if cand in df.columns:
            return cand
        low = cand.lower()
        if low in cols_lower:
            return cols_lower[low]
        # also try variants with suffixes/prefixes
        for col in df.columns:
            if col.lower().startswith(low) or col.lower().endswith(low) or low in col.lower():
                return col
    return None

def map_ohlc_columns(df):
    mapped = {}
    for key, variants in OHLC_VARIANTS.items():
        col = find_column(df, variants)
        if col:
            mapped[key] = col
    return mapped

def ensure_and_map_columns(df):
    # Detect trading/candidate date column
    date_col = find_column(df, DATE_CANDIDATES)
    if date_col is None:
        print("ERROR: No date/trading candidate column found. Columns in file:")
        print(df.columns.tolist())
        sys.exit(1)
    # Normalize date column
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.rename(columns={date_col: "candidate"})

    # Find OHLC/volume columns
    mapped = map_ohlc_columns(df)
    # If close wasn't found by map_ohlc_columns, look for other sensible names
    if "close" not in mapped:
        # try 'close' appearing anywhere
        close_col = find_column(df, ["close", "last", "settle"])
        if close_col:
            mapped["close"] = close_col

    # Show mapping result
    print("Column mapping (detected):", mapped)

    # If any of the essential columns missing, print available columns and exit
    essential = ["open", "high", "low", "close", "volume"]
    missing = [c for c in essential if c not in mapped]
    if missing:
        print("ERROR: Could not locate essential OHLC/volume columns. Missing:", missing)
        print("Available columns in CSV:", df.columns.tolist())
        # attempt to salvage: if at least 'close' exists under any name, proceed with close only
        if "close" in mapped:
            print("Proceeding with available 'close' column and filling others with NaN.")
            for k in essential:
                if k not in mapped:
                    df[k] = np.nan
        else:
            sys.exit(1)
    else:
        # rename detected columns to canonical names
        df = df.rename(columns={v: k for k, v in mapped.items()})

    # Ensure OHLC and volume exist after mapping (create if missing)
    for col in essential:
        if col not in df.columns:
            df[col] = np.nan

    # Sentiment columns
    sent_score_col = find_column(df, SENT_SCORE_VARIANTS)
    if sent_score_col:
        df = df.rename(columns={sent_score_col: "sentiment_score"})
    else:
        df["sentiment_score"] = np.nan

    sent_label_col = find_column(df, SENT_LABEL_VARIANTS)
    if sent_label_col:
        df = df.rename(columns={sent_label_col: "sentiment_label"})
    else:
        df["sentiment_label"] = np.nan

    # Normalize source/title columns if present
    if "source" not in df.columns:
        # try url or news_source
        src = find_column(df, ["source", "url", "news_source"])
        if src:
            df = df.rename(columns={src: "source"})
        else:
            df["source"] = "unknown"

    if "title" not in df.columns:
        t = find_column(df, ["title", "headline", "news_title"])
        if t:
            df = df.rename(columns={t: "title"})
        else:
            df["title"] = ""

    return df

def aggregate_daily(df):
    # We will aggregate multiple news rows that map to same candidate date
    # Prepare numeric columns
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Groupby candidate trading date
    agg = {
        "sentiment_score": ["mean", "median", "count"],
        "sentiment_label": lambda x: x.mode().iat[0] if len(x.dropna())>0 and len(x.mode())>0 else np.nan,
        "open": "first",
        "high": "first",
        "low": "first",
        "close": "first",
        "volume": "first",
        "title": lambda x: "; ".join(x.dropna().astype(str).tolist()[:3])  # brief sample titles
    }
    grouped = df.groupby("candidate").agg(agg)
    # flatten columns
    grouped.columns = ["_".join([str(i) for i in col]).strip("_") if isinstance(col, tuple) else col for col in grouped.columns.values]
    grouped = grouped.reset_index()
    # rename common names to simple ones
    rename_map = {
        "sentiment_score_mean":"sentiment_score",
        "sentiment_score_median":"sentiment_score_median",
        "sentiment_score_count":"sentiment_count",
        "sentiment_label_<lambda>":"sentiment_label",
        "title_<lambda>":"title_sample",
        "open_first":"open",
        "high_first":"high",
        "low_first":"low",
        "close_first":"close",
        "volume_first":"volume"
    }
    for k,v in rename_map.items():
        if k in grouped.columns:
            grouped = grouped.rename(columns={k:v})
    return grouped

def compute_features(df):
    # Ensure candidate is datetime and sorted
    df["candidate"] = pd.to_datetime(df["candidate"], errors="coerce")
    df = df.sort_values("candidate").reset_index(drop=True)

    # Fill missing OHLC by forward/back fill on the date index (if close only exists, other cols remain NaN)
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = df[c].ffill().bfill()

    # Basic price features
    if "close" not in df.columns:
        print("ERROR: 'close' column missing after aggregation. Columns:", df.columns.tolist())
        sys.exit(1)

    df["return_pct"] = df["close"].pct_change()
    df["log_return"] = np.log1p(df["return_pct"])
    df["next_day_return"] = df["close"].shift(-1) / df["close"] - 1
    df["next_day_dir"] = np.where(df["next_day_return"] > 0, 1, 0)

    # spreads and gaps
    df["high_low_spread"] = np.where(df["close"] != 0, (df["high"] - df["low"]) / df["close"], 0.0)
    df["open_close_gap"] = np.where(df["open"] != 0, (df["close"] - df["open"]) / df["open"], 0.0)

    # Rolling features
    for w in (3,7,14):
        df[f"ret_mean_{w}d"] = df["return_pct"].rolling(window=w, min_periods=1).mean()
        df[f"ret_std_{w}d"] = df["return_pct"].rolling(window=w, min_periods=1).std().fillna(0)

    # Lag features for sentiment_score and returns
    for lag in (1,2,3):
        df[f"sentiment_lag_{lag}"] = df["sentiment_score"].shift(lag)
        df[f"return_lag_{lag}"] = df["return_pct"].shift(lag)

    # Fill numeric NaNs with 0 for modeling (you may keep NaNs if preferred)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # Encode sentiment_label
    if "sentiment_label" in df.columns:
        df["sentiment_label"] = df["sentiment_label"].astype(str).str.lower().str.strip()
        le = LabelEncoder()
        try:
            df["sentiment_label_enc"] = le.fit_transform(df["sentiment_label"])
            joblib.dump(le, "label_encoder.joblib")
        except Exception:
            df["sentiment_label_enc"] = 0
    else:
        df["sentiment_label_enc"] = 0

    return df

def main():
    df = read_df(INPUT_CSV)
    print("Loaded columns:", df.columns.tolist())
    df = ensure_and_map_columns(df)
    print("After mapping columns, head:")
    print(df.head(3).T)

    grouped = aggregate_daily(df)
    print("Aggregated daily rows:", len(grouped))
    print("Aggregated columns:", grouped.columns.tolist())

    processed = compute_features(grouped)
    # final save
    processed.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved preprocessed dataset to: {OUTPUT_CSV}")
    print("Sample rows:")
    print(processed.head()[["candidate","sentiment_score","close","return_pct","next_day_return","next_day_dir"]])

if __name__ == "__main__":
    main()
