"""Add market-adjusted excess return to the already-matched news dataset.

excess_return_pct = stock's return_pct - KOSPI index's return_pct over the
same (base_trading_day, target_trading_day) pair. This strips out market-wide
moves (e.g. a broad rally/selloff day) so the label better reflects whether
THIS stock moved differently from the market, not just whether the whole
market was up or down that day.
"""
import pandas as pd

IN_PATH = "data/processed/news_price_matched.csv"
INDEX_PATH = "data/raw/index_history.csv"
OUT_PATH = "data/processed/news_price_matched.csv"  # overwrite with extra columns


def load_index() -> pd.Series:
    df = pd.read_csv(INDEX_PATH, index_col=0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.iloc[:, 0].sort_index()


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    index_close = load_index()

    base_days = pd.to_datetime(df["base_trading_day"])
    target_days = pd.to_datetime(df["target_trading_day"])

    index_base = index_close.reindex(base_days).values
    index_target = index_close.reindex(target_days).values
    index_return_pct = (index_target - index_base) / index_base * 100

    df["index_return_pct"] = index_return_pct.round(3)
    df["excess_return_pct"] = (df["return_pct"] - df["index_return_pct"]).round(3)

    n_missing = df["index_return_pct"].isna().sum()
    print(f"total rows: {len(df)}")
    print(f"rows missing index data: {n_missing}")

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"saved to {OUT_PATH}")

    print("\n=== excess_return_pct 기술통계 ===")
    print(df["excess_return_pct"].describe())
