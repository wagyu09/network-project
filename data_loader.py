import yfinance as yf
import pandas as pd
import requests
import random

class DataLoader:
    def __init__(self, start_date : str, end_date : str, num_stocks: int = 300):
        self.start_date = start_date
        self.end_date = end_date
        self.num_stocks = num_stocks
        self.sectors, self.tickers = self._fetch_sp500_tickers()

    def _fetch_sp500_tickers(self):
        """S&P 500 종목 티커와 섹터 정보를 Wikipedia에서 가져옵니다."""
        wiki = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = requests.get(wiki, headers={"User-Agent": "Mozilla/5.0"})
        tables = pd.read_html(req.text)
        sp500 = tables[1]
        sp500["ticker"] = sp500["Symbol"].astype(str).str.replace(".", "-", regex=False)
        
        all_tickers = sp500["ticker"].unique().tolist()
        
        # 300개 종목 랜덤 샘플링
        if len(all_tickers) > self.num_stocks:
            random.seed(42) # 재현성을 위한 시드 설정
            tickers = random.sample(all_tickers, self.num_stocks)
        else:
            tickers = all_tickers
            
        sectors = sp500[sp500['ticker'].isin(tickers)][["ticker", "GICS Sector"]]
        
        return sectors, tickers

    def load_stock_data(self):
        """S&P 500 종목들의 일별 주가 데이터를 yfinance에서 로드합니다."""
        raw = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            progress=False,
        )
        data = (
            raw.stack(0)
            .rename_axis(["date", "ticker"])
            .reset_index()[["ticker", "date", "Close", "Volume"]]
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
            .merge(self.sectors, on="ticker", how="left")
        )
        return data

    def calculate_returns(self, data):
        """주가 데이터로부터 일별 수익률과 월별 통계를 계산합니다."""
        daily = data.copy()
        daily["Daily_Return"] = daily.groupby("ticker")["Close"].pct_change()
        daily = daily.dropna(subset=["Daily_Return"])
        daily["YearMonth"] = daily["date"].dt.to_period("M")

        monthly = (
            daily.groupby(["ticker", "YearMonth"])["Daily_Return"]
            .agg(
                Monthly_Mean_Return="mean",
                Monthly_Variance="var",
                Monthly_Std_Dev="std",
            )
            .reset_index()
        )

        return daily, monthly

    def load_market_data(self):
        """S&P 500 시장 지수 데이터를 로드하고 수익률을 계산합니다."""
        mkt_idx = yf.download(
            "^GSPC",
            start=self.start_date,
            end=self.end_date,
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

    def load_data(self):
        """데이터 로딩 및 전처리 파이프라인을 실행합니다."""
        stock_data = self.load_stock_data()
        daily, monthly = self.calculate_returns(stock_data)
        mkt_idx = self.load_market_data()
        return daily, monthly, mkt_idx
