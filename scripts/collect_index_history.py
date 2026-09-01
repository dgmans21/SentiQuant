"""Pull KOSPI index history to use as a market-wide baseline for excess-return labeling.

yfinance occasionally drops a single trading day for ^KS11 specifically while individual
stock tickers have that same day fine (observed 2026-08-28, confirmed via repeated fetches
and explicit start/end -- not a transient fetch issue, the row is just absent from the
source). Individual stock prices are the source of truth for "which days were actually
trading days" (data/raw/price_history.csv), so any such isolated gap gets linearly
interpolated using that calendar. Trailing gaps (today's close not yet published) are left
as NaN rather than extrapolated, since interpolate() only fills between two known points.
"""
import os
import pandas as pd
import yfinance as yf

PRICE_PATH = "data/raw/price_history.csv"

if __name__ == "__main__":
    df = yf.download("^KS11", period="1y", interval="1d", progress=False)["Close"].squeeze()
    df.index = df.index.tz_localize(None)

    if os.path.exists(PRICE_PATH):
        prices = pd.read_csv(PRICE_PATH, index_col=0)
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        full_calendar = prices.index.union(df.index).sort_values()
        df = df.reindex(full_calendar)
        n_before_na = df.isna().sum()
        df = df.interpolate(method="linear", limit_area="inside")
        n_filled = n_before_na - df.isna().sum()
        if n_filled > 0:
            print(f"주가 거래일 기준으로 지수 데이터 {n_filled}건 보간(interpolate)함")

    df.to_csv("data/raw/index_history.csv")
    print(f"rows: {len(df)}, range: {df.index.min().date()} ~ {df.index.max().date()}")
    print("saved to data/raw/index_history.csv")
