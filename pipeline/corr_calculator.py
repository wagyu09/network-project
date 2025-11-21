"""주식 수익률의 잔차 상관관계 계산을 담당하는 모듈"""
import numpy as np
import pandas as pd

def get_residuals(daily_returns: pd.DataFrame, mkt_idx: pd.DataFrame) -> pd.DataFrame:
    """CAPM 기반으로 시장 수익률을 제거한 잔차(residual)를 계산하여 반환

    Args:
        daily_returns (pd.DataFrame): 'ticker', 'date', 'Daily_Return' 컬럼을 포함하는
            장기(long format) 일별 수익률 데이터프레임
        mkt_idx (pd.DataFrame): 'date', 'mkt_ret_spx' 컬럼을 포함하는 시장 지수 수익률 데이터프레임

    Returns:
        pd.DataFrame: 시장 효과가 제거된 잔차 수익률 데이터프레임
            데이터가 부족하거나 계산 불가 시 빈 데이터프레임 반환
    """
    if daily_returns.empty or mkt_idx.empty:
        return pd.DataFrame()

    if isinstance(mkt_idx.columns, pd.MultiIndex):
        mkt_idx.columns = mkt_idx.columns.get_level_values(0)
    
    # 주식 수익률과 시장 수익률 데이터를 날짜 기준으로 병합
    merged = daily_returns.merge(mkt_idx, on='date', how='left')[['ticker', 'date', 'Daily_Return', 'mkt_ret_spx']]

    # Wide format으로 변환하여 각 티커의 수익률을 컬럼으로, 날짜를 인덱스로 설정
    Y = merged.pivot(index='date', columns='ticker', values='Daily_Return').dropna(how='all')
    # 시장 수익률을 Y의 인덱스에 맞춰 재정렬
    m = merged.drop_duplicates('date').set_index('date')['mkt_ret_spx'].reindex(Y.index)

    # 누락된 시장 데이터가 있으면 회귀를 수행할 수 없으므로 해당 날짜를 제거
    valid_idx = m.notna()
    Y = Y[valid_idx]
    m = m[valid_idx]
    
    if Y.empty or m.isna().all():
        return pd.DataFrame()

    y = Y.values # 주식 수익률
    z = m.values.reshape(-1, 1) # 시장 수익률

    # 각 종목별로 충분한 데이터가 있는지 확인하고 회귀 수행
    y_mean = np.nanmean(y, axis=0, keepdims=True)
    z_mean = np.nanmean(z, axis=0, keepdims=True)
    
    # NaN이 아닌 값을 가진 데이터 포인트가 2개 이상인 경우에만 회귀 계산
    valid_cols = np.sum(~np.isnan(y), axis=0) >= 2
    
    y0 = y - y_mean
    z0 = z - z_mean

    # 베타(beta) 계산 (NaN 전파 방지)
    beta = np.nansum(y0[:, valid_cols] * z0, axis=0) / np.nansum(z0**2)
    
    # 알파(alpha) 계산
    alpha = y_mean[:, valid_cols] - beta * z_mean
    
    # 잔차(residuals) 계산
    E = np.full_like(y, np.nan)
    E[:, valid_cols] = y[:, valid_cols] - (alpha + z @ beta.reshape(1, -1))

    # 잔차 데이터프레임 생성 및 반환
    resid_df = pd.DataFrame(E, index=Y.index, columns=Y.columns)
    return resid_df

def calculate_residual_correlation(daily_returns: pd.DataFrame, mkt_idx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """잔차를 계산하고, 그 잔차로부터 상관관계 행렬과 통계적 유의성 엣지 목록을 반환

    Args:
        daily_returns (pd.DataFrame): 'ticker', 'date', 'Daily_Return' 컬럼을 포함하는
            장기(long format) 일별 수익률 데이터프레임
        mkt_idx (pd.DataFrame): 'date', 'mkt_ret_spx' 컬럼을 포함하는 시장 지수 수익률 데이터프레임

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - corr_matrix (pd.DataFrame): 잔차 상관관계 행렬
            - corr_stats (pd.DataFrame): 'Correlation' 및 'n'(관측치 수) 컬럼을 포함하는
              통계 정보가 담긴 엣지 목록 데이터프레임
    """
    resid_df = get_residuals(daily_returns, mkt_idx)

    if resid_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 1. 잔차 상관관계 행렬 계산
    corr_matrix = resid_df.corr()

    # 2. 통계적 유의성 계산을 위한 엣지 목록 생성
    # 상관관계 행렬을 스택하여 (티커1, 티커2, 상관관계) 형식으로 변환
    corr_data = corr_matrix.stack(future_stack=True).rename('Correlation').rename_axis(['ticker1', 'ticker2']).reset_index()
    # 중복 쌍 제거 (예: (A, B)와 (B, A) 중 하나만 남기고, (A, A) 같은 자기 자신과의 상관관계 제거)
    corr_data = corr_data.query('ticker1 < ticker2').reset_index(drop=True)

    # 관측치 수 계산 (n) - 두 티커 모두 데이터가 존재하는 날짜의 수
    n_matrix = resid_df.notna().astype(int).T @ resid_df.notna().astype(int)
    n_series = n_matrix.stack(future_stack=True).rename('n').rename_axis(['ticker1', 'ticker2']).reset_index()
    n_series = n_series.query('ticker1 < ticker2').reset_index(drop=True)
    
    # corr_data와 n_series를 'ticker1', 'ticker2' 기준으로 병합하여 엣지별 통계량 완성
    corr_stats = pd.merge(corr_data, n_series, on=['ticker1', 'ticker2'])
    # 상관관계 또는 관측치 수가 NaN인 엣지 제거
    corr_stats = corr_stats.dropna(subset=['Correlation', 'n'])

    return corr_matrix, corr_stats