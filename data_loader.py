import yfinance as yf
import pandas as pd
import requests
import random
import io
import os
import hashlib

def fetch_sp500_tickers(num_stocks: int = 500):
    """S&P 500 종목 티커와 섹터 정보를 Wikipedia에서 가져옵니다."""
    wiki = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = requests.get(wiki, headers={"User-Agent": "Mozilla/5.0"})
    tables = pd.read_html(io.StringIO(req.text), attrs={'id': 'constituents'})
    sp500 = tables[0]
    sp500["ticker"] = sp500["Symbol"].astype(str).str.replace(".", "-", regex=False)
    
    all_tickers = sp500["ticker"].unique().tolist()
    
    sectors = sp500[["ticker", "GICS Sector"]]
    
    return sectors, all_tickers

def load_raw_stock_data(tickers, start_date, end_date):
    """지정된 종목과 기간에 대한 원시 주가 데이터를 yfinance에서 로드하거나 캐시에서 불러옵니다."""
    CACHE_DIR = '.cache'
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    tickers_str = "".join(sorted(tickers))
    filename_hash = hashlib.md5(f"{tickers_str}_{start_date}_{end_date}".encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{filename_hash}.csv")

    try:
        if os.path.exists(cache_file):
            print(f"Loading data from cache: {cache_file}")
            cached_data = pd.read_csv(cache_file, header=[0, 1], index_col=0, parse_dates=True)
            # Ensure the columns are in the same order as the requested tickers
            # This can be an issue if the cached file has a different ticker order
            level_1_cols = cached_data.columns.get_level_values(1)
            if not level_1_cols.empty:
                 cached_data = cached_data.reindex(columns=tickers, level=0)
            return cached_data
    except Exception as e:
        print(f"Could not read cache file {cache_file}, re-downloading. Error: {e}")

    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )

    if not raw.empty:
        try:
            raw.to_csv(cache_file)
            print(f"Data cached to {cache_file}")
        except Exception as e:
            print(f"Failed to cache data to {cache_file}. Error: {e}")
            
    return raw

def load_market_data(start_date, end_date):
    """S&P 500 시장 지수 데이터를 로드하고 수익률을 계산합니다."""
    mkt_idx = yf.download(
        "^GSPC",
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    mkt_idx = (
        mkt_idx.reset_index()[["Date", "Close"]]
        .rename(columns={"Date": "date", "Close": "SPX"})
        .sort_values("date")
    )
    mkt_idx["mkt_ret_spx"] = mkt_idx["SPX"].pct_change()
    return mkt_idx
