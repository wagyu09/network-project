
import numpy as np
import pandas as pd

def corr_calculator(daily_returns, mkt_idx):
    if isinstance(mkt_idx.columns, pd.MultiIndex):
        mkt_idx.columns = mkt_idx.columns.get_level_values(0)
    merged = daily_returns.merge(mkt_idx, on='date', how='left')[['ticker','date','Daily_Return','mkt_ret_spx']]

    Y = merged.pivot(index='date', columns='ticker', values='Daily_Return').dropna(how='all')
    m = (
        merged.drop_duplicates('date')
              .set_index('date')['mkt_ret_spx']
              .reindex(Y.index)
      )


    y = Y.values
    z = m.values.reshape(-1, 1)

    y_mean = np.nanmean(y, axis=0, keepdims=True)
    z_mean = np.nanmean(z, axis=0, keepdims=True)
    y0 = y - y_mean
    z0 = z - z_mean

    beta  = (z0.T @ y0) / (z0.T @ z0)
    alpha = y_mean - beta * z_mean
    E = y - (alpha + z @ beta)


    resid_df    = pd.DataFrame(E, index=Y.index, columns=Y.columns)
    year_month  = resid_df.index.to_period('M')

    corr_monthly = (
        resid_df
          .groupby(year_month)
          .apply(lambda df: df.corr())
      )

    corr_monthly = corr_monthly.rename_axis(index=['YearMonth','ticker1'], columns='ticker2')

    corr_data = (
        corr_monthly
          .stack(dropna=False)          
          .rename('Correlation')
          .reset_index()
          .query('ticker1 < ticker2')  
          .set_index(['YearMonth','ticker1','ticker2'])
          .sort_index()
      )
    return corr_data, resid_df