from __future__ import annotations

from pathlib import Path

import pandas as pd


def _normalise_downloaded_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)
    return frame


def load_price_csv(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    date_column = next((column for column in frame.columns if str(column).lower() in {"date", "datetime", "timestamp"}), None)
    price_column = next((column for column in frame.columns if str(column).lower() in {"adj close", "close", "price"}), None)
    if date_column is None or price_column is None:
        raise ValueError("CSV must contain a date column and one of: Adj Close, Close, Price.")
    result = frame[[date_column, price_column]].rename(columns={date_column: "Date", price_column: "price"})
    result["Date"] = pd.to_datetime(result["Date"], utc=True).dt.tz_localize(None)
    result["price"] = pd.to_numeric(result["price"], errors="coerce")
    result = result.dropna().drop_duplicates("Date").sort_values("Date").set_index("Date")
    if len(result) < 30 or (result["price"] <= 0).any():
        raise ValueError("The price series must contain at least 30 positive observations.")
    return result


def download_prices(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted daily prices through yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install the data extra with: pip install -e '.[full]'") from exc
    raw = yf.download(tickers=ticker, start=start, end=end, auto_adjust=True, progress=False, multi_level_index=False)
    raw = _normalise_downloaded_columns(raw)
    if raw is None or raw.empty:
        raise ValueError(f"No market data returned for {ticker}.")
    price_column = "Close" if "Close" in raw.columns else "Adj Close"
    result = raw[[price_column]].rename(columns={price_column: "price"}).reset_index()
    result["Date"] = pd.to_datetime(result["Date"], utc=True).dt.tz_localize(None)
    result = result.set_index("Date").sort_index()
    result["price"] = pd.to_numeric(result["price"], errors="coerce")
    return result.dropna()


def save_price_csv(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.rename(columns={"price": "Close"}).to_csv(destination, index_label="Date")

