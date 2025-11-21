import pandas as pd
import numpy as np

def get_portfolio_performance(portfolio_prices: pd.DataFrame):
    """Calculates key performance metrics for a given portfolio price series."""
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

    daily_returns_portfolio = daily_returns_individual.mean(axis=1)
    
    cumulative_return = (1 + daily_returns_portfolio).prod() - 1
    volatility = daily_returns_portfolio.std() * np.sqrt(252)
    sharpe_ratio = (daily_returns_portfolio.mean() * np.sqrt(252) / volatility) if volatility != 0 else np.nan
    
    cumulative_asset_value = (1 + daily_returns_portfolio).cumprod()
    peak = cumulative_asset_value.expanding(min_periods=1).max()
    drawdown = (cumulative_asset_value / peak) - 1
    max_drawdown = drawdown.min()
    
    return {
        'Cumulative_Return': cumulative_return, 'Volatility': volatility,
        'Sharpe_Ratio': sharpe_ratio, 'Max_Drawdown': max_drawdown
    }
