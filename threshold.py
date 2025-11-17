from scipy import stats
import numpy as np
import pandas as pd

def threshold(corr_data, resid_df, alpha = 0.05, c_min = 0):
    n_tbl = (
        resid_df.notna().astype(int)
        .groupby(resid_df.index.to_period('M'))
        .apply(lambda m: pd.DataFrame(m.T @ m, index=m.columns, columns=m.columns))
        .rename_axis(index=['YearMonth', 'ticker1'], columns='ticker2')
        .stack()
        .rename('n')
        .reset_index()
        .query('ticker1 < ticker2')
        .set_index(['YearMonth', 'ticker1', 'ticker2'])
    )


    df = corr_data.join(n_tbl, how='inner').dropna(subset=['Correlation', 'n'])
    r = df['Correlation'].to_numpy(float)
    n = df['n'].to_numpy(int)

    dfree = n - 2 - 1                        #
    r_safe = np.clip(r, -0.999999, 0.999999)
    with np.errstate(divide='ignore', invalid='ignore'):
        tval = r_safe * np.sqrt(np.where(dfree > 0, dfree, np.nan) / (1.0 - r_safe**2))
    pval = 2.0 * stats.t.sf(np.abs(tval), df=np.where(dfree > 0, dfree, 1))
    pval[(dfree < 1) | ~np.isfinite(pval)] = 1.0

    df['pval'] = pval

    p_edges = df[(df['pval'] < alpha) & (df['Correlation'].abs() >= c_min)].sort_index()
    
    return p_edges