# merge_sentiment_nifty.py
"""
Usage:
  1. Put your sentiment CSV in the same folder and update SENTIMENT_CSV if necessary.
  2. Run: python merge_sentiment_nifty.py
  3. Output: merged_sentiment_nifty.csv

This version is defensive against yfinance returning:
 - MultiIndex columns (ticker, field)
 - single-level duplicated ticker column names (e.g. ['^NSEI','^NSEI',...])
 - missing 'Adj Close' column
It will attempt to recover column names by position if it detects duplicated ticker labels.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import sys

# --------- USER CONFIG ----------
SENTIMENT_CSV = "output_news/nifty_news_gdelt_raw.csv"   # change if your file has another name
OUTPUT_CSV = "merged_sentiment_nifty.csv"
NIFTY_TICKER = "^NSEI"   # Yahoo ticker for Nifty 50
DATE_COL = "date"        # the column in your sentiment CSV with dates
# --------------------------------

# column alias mappings for sentiment file
SENTIMENT_ALIASES = {
    "sentiment_score": ["sentiment_score", "sent_score", "score", "sentiment"],
    "sentiment_label": ["sentiment_label", "sent_label", "label"],
    "source": ["source", "url", "news_source"],
    "title": ["title", "headline", "news_title"]
}

def find_first_available(col_list, df_cols):
    for c in col_list:
        if c in df_cols:
            return c
    return None

def load_sentiment(path):
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if DATE_COL not in df.columns:
        raise SystemExit(f"Date column '{DATE_COL}' not found in {path}. Found columns: {df.columns.tolist()}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    if df[DATE_COL].isnull().any():
        n_bad = df[DATE_COL].isnull().sum()
        print(f"Warning: {n_bad} rows have unparsable dates and will be dropped.")
        df = df[df[DATE_COL].notnull()].copy()
    df["orig_date"] = df[DATE_COL]

    # detect and rename common sentiment columns
    cols = df.columns.tolist()
    rename_map = {}
    score_col = find_first_available(SENTIMENT_ALIASES["sentiment_score"], cols)
    label_col = find_first_available(SENTIMENT_ALIASES["sentiment_label"], cols)
    src_col = find_first_available(SENTIMENT_ALIASES["source"], cols)
    title_col = find_first_available(SENTIMENT_ALIASES["title"], cols)
    if score_col: rename_map[score_col] = "sentiment_score"
    if label_col: rename_map[label_col] = "sentiment_label"
    if src_col: rename_map[src_col] = "source"
    if title_col: rename_map[title_col] = "title"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def fetch_nifty_ohlc(start_date, end_date, ticker=NIFTY_TICKER):
    start_fetch = (start_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_fetch = (end_date + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    print(f"Fetching {ticker} from {start_fetch} to {end_fetch} ...")
    # request with auto_adjust=False to keep 'Adj Close' when available and silence the FutureWarning
    data = yf.download(ticker, start=start_fetch, end=end_fetch, progress=False, auto_adjust=False)

    if data.empty:
        raise RuntimeError("No data returned from yfinance. Check ticker or internet connection.")

    # --- Handle MultiIndex columns (ticker, field) by dropping top level if present ---
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # often the top level is the ticker; dropping it gives columns like Open, High, ...
            data.columns = data.columns.droplevel(0)
            print("Info: Dropped top level from MultiIndex columns returned by yfinance.")
        except Exception:
            # fallback: flatten by joining levels with underscore
            data.columns = ['_'.join(map(str, c)).strip() for c in data.columns.values]
            print("Info: Flattened MultiIndex columns by joining levels.")

    # --- Handle the case where yfinance returned duplicated ticker names as column labels ---
    # Example observed: data.columns == ['^NSEI','^NSEI','^NSEI','^NSEI','^NSEI','^NSEI']
    col_list = list(data.columns)
    if len(col_list) >= 5 and all((str(c) == str(ticker) for c in col_list)):
        # assume the column order is the usual yfinance order:
        # Open, High, Low, Close, Adj Close, Volume  (length may be 5 or 6)
        print("Warning: yfinance returned duplicated ticker column names. Attempting to assign standard OHLC column names by position.")
        possible_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        # take as many as available by position
        assigned = possible_cols[:len(col_list)]
        data.columns = assigned
        print(f"Assigned column names: {assigned}")

    # After above fixes, select present OHLC-like columns
    possible_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    present = [c for c in possible_cols if c in data.columns]
    if not present:
        # as a last resort, try case-insensitive match or pprint columns for debugging
        # show user the columns and raise an informative error
        raise RuntimeError(f"No OHLC/Volume columns found in downloaded data. Columns present: {data.columns.tolist()}")

    data = data[present].copy()

    # rename to canonical lowercase names when present
    rename_map = { "Open":"open", "High":"high", "Low":"low", "Close":"close", "Adj Close":"adj_close", "Volume":"volume" }
    rename_map = {k:v for k,v in rename_map.items() if k in data.columns}
    data = data.rename(columns=rename_map)

    # normalize index to dates only (no time component)
    data.index = pd.to_datetime(data.index).normalize()
    data = data[~data.index.duplicated(keep='first')]
    return data

def find_next_trading_date(target_date, trading_index):
    td = pd.to_datetime(target_date).normalize()
    pos = trading_index.searchsorted(td)
    if pos >= len(trading_index):
        return pd.NaT
    return trading_index[pos]

def merge_sentiment_with_ohlc(sent_df, nifty_df):
    trading_dates = nifty_df.index.sort_values()
    sent_df[DATE_COL] = pd.to_datetime(sent_df[DATE_COL]).dt.normalize()

    mapped_dates = []
    trading_flags = []
    for d in sent_df[DATE_COL]:
        if d in trading_dates:
            mapped_dates.append(d)
            trading_flags.append(True)
        else:
            nd = find_next_trading_date(d, trading_dates)
            mapped_dates.append(nd)
            trading_flags.append(False)

    sent_df["trading_day"] = trading_flags
    sent_df["trading_date"] = pd.to_datetime(mapped_dates)

    if sent_df["trading_date"].isnull().any():
        missing_count = int(sent_df["trading_date"].isnull().sum())
        print(f"Warning: {missing_count} sentiment rows could not be mapped to any future trading date. They will be dropped.")
        sent_df = sent_df[sent_df["trading_date"].notnull()].copy()

    nifty_reset = nifty_df.reset_index()
    first_col = nifty_reset.columns[0]
    nifty_reset = nifty_reset.rename(columns={first_col: "trading_date"})
    nifty_reset["trading_date"] = pd.to_datetime(nifty_reset["trading_date"]).dt.normalize()

    if isinstance(nifty_reset.columns, pd.MultiIndex):
        nifty_reset.columns = ['_'.join(map(str, c)).strip() for c in nifty_reset.columns.values]

    # Merge now
    merged = pd.merge(
        sent_df,
        nifty_reset,
        how="left",
        on="trading_date",
        sort=False,
        validate="m:1"
    )

    # Normalize date column name to original date
    if "orig_date" in merged.columns:
        merged = merged.rename(columns={"orig_date": "date"})
    elif DATE_COL in merged.columns and DATE_COL != "date":
        merged = merged.rename(columns={DATE_COL: "date"})

    merged["candidate"] = merged["trading_date"]

    final_cols = ["date", "source", "title", "sentiment_label", "sentiment_score",
                  "open", "high", "low", "close", "volume", "trading_day", "candidate"]

    for c in final_cols:
        if c not in merged.columns:
            merged[c] = np.nan

    result = merged[final_cols].copy()
    return result

def main():
    try:
        sentiment = load_sentiment(SENTIMENT_CSV)
    except Exception as e:
        print("Error loading sentiment CSV:", e)
        sys.exit(1)

    if sentiment.empty:
        print("No sentiment rows found after parsing dates. Exiting.")
        sys.exit(0)

    min_date = sentiment[DATE_COL].min().normalize()
    max_date = sentiment[DATE_COL].max().normalize()

    try:
        nifty = fetch_nifty_ohlc(min_date, max_date)
    except Exception as e:
        print("Error fetching Nifty OHLC:", e)
        sys.exit(1)

    merged = merge_sentiment_with_ohlc(sentiment, nifty)

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved merged file to: {OUTPUT_CSV}")
    print("Done.")

if __name__ == "__main__":
    main()
