import os
import sys
import pandas as pd
import io
import networkx as nx
import numpy as np
import random
import json

# Add the parent directory to the system path to allow module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import fetch_sp500_tickers, load_raw_stock_data, DataLoader
from corr_calculator import get_residuals, calculate_residual_correlation
from threshold import threshold
from community_detection import create_network_from_edges, detect_communities, calculate_inter_community_correlation
from network_visualizer import visualize_network

def get_quarter_dates(quarter):
    """Converts a pandas Quarter object to start and end date strings."""
    start_date = quarter.start_time.strftime('%Y-%m-%d')
    end_date = quarter.end_time.strftime('%Y-%m-%d')
    return start_date, end_date

def calculate_centrality(G: nx.Graph):
    """
    주어진 그래프의 각 노드에 대한 연결 중심성(Degree Centrality)을 계산합니다.
    """
    if not G.nodes():
        return {}
    return nx.degree_centrality(G)

def get_portfolio_performance(portfolio_prices: pd.DataFrame):
    """
    주어진 포트폴리오 가격 데이터로부터 일별 수익률을 계산하고 주요 성과 지표를 반환합니다.
    (동일 가중 포트폴리오 가정)
    """
    if portfolio_prices.empty:
        return {
            'Cumulative_Return': np.nan,
            'Volatility': np.nan,
            'Sharpe_Ratio': np.nan,
            'Max_Drawdown': np.nan
        }
    
    # 각 종목의 일별 수익률 계산
    daily_returns_individual = portfolio_prices.pct_change().dropna()
    
    if daily_returns_individual.empty:
        return {
            'Cumulative_Return': np.nan,
            'Volatility': np.nan,
            'Sharpe_Ratio': np.nan,
            'Max_Drawdown': np.nan
        }

    # 동일 가중 포트폴리오의 일별 수익률 (각 종목의 수익률을 평균)
    daily_returns_portfolio = daily_returns_individual.mean(axis=1)
    
    # 누적 수익률 계산
    cumulative_return = (1 + daily_returns_portfolio).prod() - 1

    # 변동성 (연율화 표준편차, 252 거래일 기준)
    volatility = daily_returns_portfolio.std() * np.sqrt(252)

    # 샤프 비율 (무위험 수익률은 0으로 가정)
    sharpe_ratio = daily_returns_portfolio.mean() * np.sqrt(252) / volatility if volatility != 0 else np.nan

    # 최대 낙폭 (Maximum Drawdown) 계산
    cumulative_asset_value = (1 + daily_returns_portfolio).cumprod()
    peak = cumulative_asset_value.expanding(min_periods=1).max()
    drawdown = (cumulative_asset_value / peak) - 1
    max_drawdown = drawdown.min()
    
    return {
        'Cumulative_Return': cumulative_return,
        'Volatility': volatility,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown
    }

def run_pipeline(num_stocks=300, alpha=0.01):
    """
    Executes the entire rolling window pipeline.

    :param num_stocks: Number of stocks to sample from S&P 500.
    :param alpha: Significance level for edge filtering.
    """
    all_quarters = pd.period_range(start='2017Q3', end='2025Q3', freq='Q')
    
    print(f"Fetching {num_stocks} S&P 500 tickers...")
    try:
        sectors, initial_tickers = fetch_sp500_tickers(num_stocks=num_stocks)
        print("Ticker fetching complete.")
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return

    num_test_sets = len(all_quarters) - 3

    for i in range(num_test_sets):
        network_quarters = all_quarters[i:i+3]
        test_quarter = all_quarters[i+3]
        full_period = all_quarters[i:i+4]
        
        full_period_start, _ = get_quarter_dates(full_period[0])
        _, full_period_end = get_quarter_dates(full_period[-1])

        folder_name = f"Test_{i+1:02d}_({full_period[0]}-{full_period[-1]})"
        output_dir = os.path.join(os.path.dirname(__file__), 'tests', folder_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n--- Processing: {folder_name} ---")

        print("  Loading data for the full 4-quarter period to validate tickers...")
        full_data = load_raw_stock_data(initial_tickers, full_period_start, full_period_end)
        
        if full_data.empty:
            print(f"  Warning: No data loaded for period {full_period_start} to {full_period_end}. Skipping.")
            continue
        
        close_prices_all = full_data.xs('Close', level=1, axis=1) if isinstance(full_data.columns, pd.MultiIndex) else full_data.get('Close')
        if close_prices_all is None:
            print(f"  Warning: 'Close' data not found for period {full_period_start} to {full_period_end}. Skipping.")
            continue
            
        valid_tickers = close_prices_all.dropna(axis=1).columns.tolist()

        if not valid_tickers:
            print("  Warning: No valid tickers found for this entire period. Skipping.")
            continue
            
        print(f"  Found {len(valid_tickers)} tickers with complete data for the 4-quarter period.")

        test_start_date, test_end_date = get_quarter_dates(test_quarter)
        network_start_date, network_end_date = get_quarter_dates(network_quarters[0])[0], get_quarter_dates(network_quarters[-1])[1]
        
        # DataLoader for network construction
        network_data_loader = DataLoader(start_date=network_start_date, end_date=network_end_date, num_stocks=len(valid_tickers))
        network_data_loader.tickers = valid_tickers
        network_data_loader.sectors = sectors[sectors['ticker'].isin(valid_tickers)]
        network_daily_returns, _, network_mkt_idx = network_data_loader.load_data()
        
        # DataLoader for test period (to get raw returns for backtesting)
        test_data_loader = DataLoader(start_date=test_start_date, end_date=test_end_date, num_stocks=len(valid_tickers))
        test_data_loader.tickers = valid_tickers
        test_data_loader.sectors = sectors[sectors['ticker'].isin(valid_tickers)]
        test_daily_returns, _, test_mkt_idx = test_data_loader.load_data() # Need this for market_data in residuals but not for raw returns
        
        # Get raw test data prices for backtesting (from load_raw_stock_data output)
        raw_test_data_prices = load_raw_stock_data(valid_tickers, test_start_date, test_end_date)
        test_prices_for_performance = raw_test_data_prices.xs('Close', level=1, axis=1)


        print("  Calculating residuals for network construction...")
        corr_matrix, corr_stats = calculate_residual_correlation(network_daily_returns, network_mkt_idx)
        # test_residuals_df = get_residuals(test_daily_returns, test_mkt_idx) # Not needed for raw return backtest

        # Save raw test data for valid tickers
        raw_data_path = os.path.join(output_dir, 'raw_test_data.csv')
        raw_test_data_prices.to_csv(raw_data_path) # Save original full raw data
        print(f"  Raw test data for {len(valid_tickers)} tickers saved to {raw_data_path}")

        if corr_matrix.empty:
            print("  Warning: Correlation matrix is empty. Skipping artifact generation.")
            continue
            
        corr_matrix_path = os.path.join(output_dir, 'correlation_matrix.csv')
        corr_matrix.to_csv(corr_matrix_path)
        print(f"  Full correlation matrix saved to {corr_matrix_path}")

        print("  Filtering edges with alpha=0.01 and c_min=0...")
        p_edges = threshold(corr_stats, alpha=alpha, c_min=0.0)
        
        G_for_centrality = create_network_from_edges(p_edges, weight_col='Correlation')
        G_for_centrality.add_nodes_from(valid_tickers)

        if p_edges.empty:
            print("  Warning: No significant edges found.")
            partition = {node: i for i, node in enumerate(valid_tickers)}
            inter_community_matrix = pd.DataFrame()
        else:
            print(f"  Found {len(p_edges)} significant edges.")
            positive_edges = p_edges[p_edges['Correlation'] > 0]
            if positive_edges.empty:
                print("  Warning: No positive edges for community detection.")
                partition = {node: i for i, node in enumerate(valid_tickers)}
                inter_community_matrix = pd.DataFrame()
            else:
                G_community = create_network_from_edges(positive_edges, weight_col='Correlation')
                G_community.add_nodes_from(valid_tickers)
                partition = detect_communities(G_community, weight_col='Correlation', random_state=42)
                print(f"  Detected {len(set(partition.values()))} communities.")
                inter_community_matrix = calculate_inter_community_correlation(p_edges, partition)
        
        inter_comm_path = os.path.join(output_dir, 'inter_community_correlation.csv')
        inter_community_matrix.to_csv(inter_comm_path)
        print(f"  Inter-community correlation matrix saved to {inter_comm_path}")

        print("  Constructing portfolios...")
        centrality = calculate_centrality(G_for_centrality)
        portfolios = {'min_centrality_portfolio': [], 'max_centrality_portfolio': [], 'random_portfolio': []}
        
        communities_to_tickers = {comm_id: [t for t, c in partition.items() if c == comm_id and t in valid_tickers] for comm_id in set(partition.values())}
        communities_to_tickers = {k: v for k, v in communities_to_tickers.items() if v}
        
        min_centrality_selection = []
        max_centrality_selection = []

        for comm_id, tickers_in_comm in communities_to_tickers.items():
            filtered_centrality = {t: centrality.get(t, 0) for t in tickers_in_comm}
            if not filtered_centrality: continue
            min_centrality_selection.append(min(filtered_centrality, key=filtered_centrality.get))
            max_centrality_selection.append(max(filtered_centrality, key=filtered_centrality.get))

        portfolios['min_centrality_portfolio'] = min_centrality_selection
        portfolios['max_centrality_portfolio'] = max_centrality_selection
        
        num_stocks_to_select = len(communities_to_tickers)
        if num_stocks_to_select > 0 and len(valid_tickers) >= num_stocks_to_select:
            portfolios['random_portfolio'] = random.sample(valid_tickers, num_stocks_to_select)

        portfolios_path = os.path.join(output_dir, 'portfolios.json')
        with open(portfolios_path, 'w') as f: json.dump(portfolios, f, indent=4)
        print(f"  Portfolio compositions saved to {portfolios_path}")

        print("  Calculating portfolio performance...") # Removed "on RESIDUALS"
        all_performance_results = {}
        for p_name, p_tickers in portfolios.items():
            if not p_tickers or test_prices_for_performance.empty:
                performance = {}
            else:
                portfolio_price_data = test_prices_for_performance[p_tickers]
                performance = get_portfolio_performance(portfolio_price_data) # Pass prices
            all_performance_results[p_name] = performance
            
        performance_df = pd.DataFrame.from_dict(all_performance_results, orient='index')
        performance_path = os.path.join(output_dir, 'backtest_results.csv')
        performance_df.to_csv(performance_path)
        print(f"  Portfolio performance results saved to {performance_path}")

        G_final_visual = create_network_from_edges(p_edges, weight_col='Correlation', edge_attrs=['Correlation'])
        G_final_visual.add_nodes_from(valid_tickers)
        viz_path = os.path.join(output_dir, 'network_visualization.png')
        print("  Generating and saving network visualization...")
        visualize_network(G_final_visual, partition, output_filename=viz_path)

        print(f"--- Completed: {folder_name} ---")

if __name__ == '__main__':
    run_pipeline(num_stocks=300)