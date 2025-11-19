import numpy as np
import pandas as pd

def get_residuals(daily_returns, mkt_idx):
    """
    CAPM 기반으로 시장 수익률을 제거한 잔차(residual)를 계산하여 반환합니다.
    """
    if daily_returns.empty or mkt_idx.empty:
        return pd.DataFrame()

    if isinstance(mkt_idx.columns, pd.MultiIndex):
        mkt_idx.columns = mkt_idx.columns.get_level_values(0)
    
    merged = daily_returns.merge(mkt_idx, on='date', how='left')[['ticker', 'date', 'Daily_Return', 'mkt_ret_spx']]

    Y = merged.pivot(index='date', columns='ticker', values='Daily_Return').dropna(how='all')
    m = merged.drop_duplicates('date').set_index('date')['mkt_ret_spx'].reindex(Y.index)

    # 누락된 시장 데이터가 있으면 회귀를 수행할 수 없으므로 해당 날짜를 제거
    valid_idx = m.notna()
    Y = Y[valid_idx]
    m = m[valid_idx]
    
    if Y.empty or m.isna().all():
        return pd.DataFrame()

    y = Y.values
    z = m.values.reshape(-1, 1)
    
    # 각 종목별로 충분한 데이터가 있는지 확인하고 회귀 수행
    y_mean = np.nanmean(y, axis=0, keepdims=True)
    z_mean = np.nanmean(z, axis=0, keepdims=True)
    
    # NaN이 아닌 값을 가진 데이터 포인트가 2개 이상인 경우에만 회귀 계산
    valid_cols = np.sum(~np.isnan(y), axis=0) >= 2
    
    y0 = y - y_mean
    z0 = z - z_mean

    # beta 계산 (NaN 전파 방지)
    beta = np.nansum(y0[:, valid_cols] * z0, axis=0) / np.nansum(z0**2)
    
    alpha = y_mean[:, valid_cols] - beta * z_mean
    
    E = np.full_like(y, np.nan)
    E[:, valid_cols] = y[:, valid_cols] - (alpha + z @ beta.reshape(1, -1))

    resid_df = pd.DataFrame(E, index=Y.index, columns=Y.columns)
    return resid_df

def calculate_residual_correlation(daily_returns, mkt_idx):
    """
    잔차를 계산하고, 그 잔차로부터 상관관계 행렬과 통계적 유의성 엣지 목록을 반환합니다.
    """
    resid_df = get_residuals(daily_returns, mkt_idx)

    if resid_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 1. 전체 기간에 대한 상관관계 행렬 계산
    corr_matrix = resid_df.corr()

    # 2. 통계적 유의성 계산을 위한 엣지 목록 생성
    corr_data = corr_matrix.stack(dropna=False).rename('Correlation').rename_axis(['ticker1', 'ticker2']).reset_index()
    corr_data = corr_data.query('ticker1 < ticker2').reset_index(drop=True)

    # 관측치 수 계산 (n)
    n_matrix = resid_df.notna().astype(int).T @ resid_df.notna().astype(int)
    n_series = n_matrix.stack().rename('n').rename_axis(['ticker1', 'ticker2']).reset_index()
    n_series = n_series.query('ticker1 < ticker2').reset_index(drop=True)
    
    # corr_data와 n_series를 'ticker1', 'ticker2' 기준으로 병합
    corr_stats = pd.merge(corr_data, n_series, on=['ticker1', 'ticker2'])
    corr_stats = corr_stats.dropna(subset=['Correlation', 'n'])

    return corr_matrix, corr_stats
