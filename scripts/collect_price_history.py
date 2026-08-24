"""Pull enough price history to cover the full news collection window (Phase 1 backfill
goes back to ~April 2026 for some stocks), so every news item has a price to match against.
"""
import pandas as pd
import yfinance as yf

from collect_multi_sample import STOCKS

if __name__ == "__main__":
    tickers = [ticker for ticker, _ in STOCKS.values()]
    df = yf.download(tickers, period="1y", interval="1d", progress=False)["Close"]
    df.to_csv("data/raw/price_history.csv")
    print(f"rows: {len(df)}, range: {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"saved to data/raw/price_history.csv")
