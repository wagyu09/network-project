
import numpy as np
import pandas as pd

def _calculate_residuals(daily_returns, mkt_idx):
    """CAPM 기반으로 시장 수익률을 제거한 잔차(residual)를 계산합니다."""
    if isinstance(mkt_idx.columns, pd.MultiIndex):
        mkt_idx.columns = mkt_idx.columns.get_level_values(0)
    merged = daily_returns.merge(mkt_idx, on='date', how='left')[['ticker', 'date', 'Daily_Return', 'mkt_ret_spx']]

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
    year_quarter = resid_df.index.to_period('Q')

    corr_quarterly = (
        resid_df
          .groupby(year_quarter)
          .apply(lambda df: df.corr())
      )

    corr_quarterly = corr_quarterly.rename_axis(index=['YearQuarter','ticker1'], columns='ticker2')

    corr_data = (
        corr_quarterly
          .stack(dropna=False)          
          .rename('Correlation')
          .reset_index()
          .query('ticker1 < ticker2')  
          .set_index(['YearQuarter','ticker1','ticker2'])
          .sort_index()
      )
    n_tbl = (
        resid_df.notna().astype(int)
        .groupby(resid_df.index.to_period('Q'))
        .apply(lambda m: pd.DataFrame(m.T @ m, index=m.columns, columns=m.columns))
        .rename_axis(index=['YearQuarter', 'ticker1'], columns='ticker2')
        .stack()
        .rename('n')
        .reset_index()
        .query('ticker1 < ticker2')
        .set_index(['YearQuarter', 'ticker1', 'ticker2'])
    )

    corr_stats = corr_data.join(n_tbl, how='inner').dropna(subset=['Correlation', 'n'])

    return corr_stats