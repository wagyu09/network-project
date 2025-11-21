"""포트폴리오 성과 측정과 관련된 함수들을 관리하는 모듈"""
import pandas as pd
import numpy as np

def get_portfolio_performance(portfolio_prices: pd.DataFrame):
    """특정 포트폴리오의 가격 데이터로부터 주요 성과 지표를 계산

    동일 가중(Equal-weighted) 포트폴리오를 가정하고, 개별 자산 수익률의
    일별 평균을 포트폴리오의 일별 수익률로 사용함

    Args:
        portfolio_prices (pd.DataFrame): 인덱스는 날짜(datetime), 컬럼은 티커,
            값은 종가인 데이터프레임

    Returns:
        dict: 누적 수익률, 연율화 변동성, 샤프 지수, 최대 낙폭(MDD)을
            포함하는 딕셔너리. 데이터가 부족해 계산이 불가능할 경우
            모든 값은 NaN으로 반환
    """
    if portfolio_prices.empty or portfolio_prices.isnull().all().all():
        return {
            'Cumulative_Return': np.nan, 'Volatility': np.nan,
            'Sharpe_Ratio': np.nan, 'Max_Drawdown': np.nan
        }
    
    daily_returns_individual = portfolio_prices.pct_change(fill_method=None).dropna()
    if daily_returns_individual.empty:
        return {
            'Cumulative_Return': np.nan, 'Volatility': np.nan,
            'Sharpe_Ratio': np.nan, 'Max_Drawdown': np.nan
        }

    # 동일 가중 포트폴리오의 일별 수익률
    daily_returns_portfolio = daily_returns_individual.mean(axis=1)
    
    # 1. 누적 수익률 계산
    cumulative_return = (1 + daily_returns_portfolio).prod() - 1
    # 2. 연율화 변동성 계산 (일별 수익률의 표준편차 * sqrt(252))
    volatility = daily_returns_portfolio.std() * np.sqrt(252)
    # 3. 샤프 지수 계산 (연율화된 (수익률/변동성), 무위험 수익률은 0으로 가정)
    sharpe_ratio = (daily_returns_portfolio.mean() * np.sqrt(252) / volatility) if volatility != 0 else np.nan
    
    # 4. 최대 낙폭 (Max Drawdown) 계산
    cumulative_asset_value = (1 + daily_returns_portfolio).cumprod()
    peak = cumulative_asset_value.expanding(min_periods=1).max()
    drawdown = (cumulative_asset_value / peak) - 1
    max_drawdown = drawdown.min()
    
    return {
        'Cumulative_Return': cumulative_return, 'Volatility': volatility,
        'Sharpe_Ratio': sharpe_ratio, 'Max_Drawdown': max_drawdown
    }