from pathlib import Path
import pandas as pd


def fetch_stooq_daily(symbol: str) -> pd.DataFrame:

    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
    df = pd.read_csv(url)
    return df

def main():
    symbol = "XOM"

    df = fetch_stooq_daily(symbol)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    out_dir = Path("data/raw/stooq")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{symbol}_daily.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved Stooq daily data to: {out_path}")
    print(df.tail())


if __name__ == "__main__":
    main()