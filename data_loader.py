import yfinance as yf
import pandas as pd

class DataLoader:
    def __init__(self, start_date : str, end_date : str):
        self.start_date = start_date
        self.end_date = end_date

        wiki = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500 = pd.read_html(wiki, storage_options={"User-Agent": "Mozilla/5.0"})[1]
        sp500["ticker"] = sp500["Symbol"].astype(str).str.replace(".", "-", regex=False)

        self.sectors = sp500[["ticker", "GICS Sector"]]
        self.tickers = self.sectors["ticker"].unique().tolist()

    def load_data(self):
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

        return daily, monthly, mkt_idx
