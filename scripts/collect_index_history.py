"""Pull KOSPI index history to use as a market-wide baseline for excess-return labeling."""
import yfinance as yf

if __name__ == "__main__":
    df = yf.download("^KS11", period="1y", interval="1d", progress=False)["Close"]
    df.to_csv("data/raw/index_history.csv")
    print(f"rows: {len(df)}, range: {df.index.min().date()} ~ {df.index.max().date()}")
    print("saved to data/raw/index_history.csv")
