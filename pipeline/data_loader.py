# 주가 데이터 및 관련 정보를 로드하고 전처리하는 모듈
import yfinance as yf
import pandas as pd
import requests
import random
import io
import os
import hashlib
from config import EXCLUDED_TICKERS

def fetch_sp500_tickers(num_stocks: int = 500) -> tuple[pd.DataFrame, list]:
    """S&P 500 종목 티커와 섹터 정보를 Wikipedia에서 가져옴

    Args:
        num_stocks (int): S&P 500 종목 중 가져올 최대 개수 (현재 사용되지 않음)

    Returns:
        tuple[pd.DataFrame, list]:
            - sectors (pd.DataFrame): 티커와 GICS 섹터 정보를 담은 데이터프레임
            - all_tickers (list): 제외 종목 필터링 후의 티커 리스트
    """
    wiki = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = requests.get(wiki, headers={"User-Agent": "Mozilla/5.0"})
    tables = pd.read_html(io.StringIO(req.text), attrs={'id': 'constituents'})
    sp500 = tables[0]
    sp500["ticker"] = sp500["Symbol"].astype(str).str.replace(".", "-", regex=False)
    
    all_tickers = sp500["ticker"].unique().tolist()
    
    # config.EXCLUDED_TICKERS에 정의된 종목들을 리스트에서 제거
    for i in EXCLUDED_TICKERS:
        if i in all_tickers:
            all_tickers.remove(i)
        else:
            pass
            
    sectors = sp500[sp500['ticker'].isin(all_tickers)][["ticker", "GICS Sector"]]
    
    return sectors, all_tickers

def load_raw_stock_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """지정된 종목과 기간에 대한 원시 주가 데이터를 yfinance에서 로드하거나 캐시에서 불러옴

    Args:
        tickers (list): 주가 데이터를 로드할 종목 티커 리스트
        start_date (str): 데이터 로드 시작일 ('YYYY-MM-DD' 형식)
        end_date (str): 데이터 로드 종료일 ('YYYY-MM-DD' 형식)

    Returns:
        pd.DataFrame: 로드된 원시 주가 데이터 (멀티 인덱스 컬럼 구조)
            데이터 로드에 실패하면 빈 데이터프레임 반환
    """
    CACHE_DIR = '.cache'
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 캐시 파일명 생성을 위한 해시값 생성
    tickers_str = "".join(sorted(tickers))
    filename_hash = hashlib.md5(f"{tickers_str}_{start_date}_{end_date}".encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{filename_hash}.csv")

    # 캐시 파일이 존재하면 불러오고, 없으면 yfinance에서 다운로드
    try:
        if os.path.exists(cache_file):
            print(f"Loading data from cache: {cache_file}")
            cached_data = pd.read_csv(cache_file, header=[0, 1], index_col=0, parse_dates=True)
            # 요청된 티커 순서와 캐시 파일의 컬럼 순서를 맞춤
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

    # 다운로드된 데이터가 있으면 캐시 파일로 저장
    if not raw.empty:
        try:
            raw.to_csv(cache_file)
            print(f"Data cached to {cache_file}")
        except Exception as e:
            print(f"Failed to cache data to {cache_file}. Error: {e}")
            
    return raw

def load_market_data(start_date: str, end_date: str) -> pd.DataFrame:
    """S&P 500 시장 지수 데이터를 로드하고 수익률을 계산함

    Args:
        start_date (str): 데이터 로드 시작일 ('YYYY-MM-DD' 형식)
        end_date (str): 데이터 로드 종료일 ('YYYY-MM-DD' 형식)
    Returns:
        pd.DataFrame: S&P 500 지수 데이터와 일별 시장 수익률 ('mkt_ret_spx') 컬럼
    """
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