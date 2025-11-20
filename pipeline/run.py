import os
import sys
import pandas as pd
import numpy as np
import random
import json
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import fetch_sp500_tickers, load_raw_stock_data, load_market_data
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
    """Calculates the Degree Centrality for each node in the graph."""
    if not G.nodes():
        return {}
    return nx.degree_centrality(G)

def get_portfolio_performance(portfolio_prices: pd.DataFrame):
    """Calculates key performance metrics for a given portfolio price series."""
    if portfolio_prices.empty or portfolio_prices.isnull().all().all():
        return {
            'Cumulative_Return': np.nan, 'Volatility': np.nan,
            'Sharpe_Ratio': np.nan, 'Max_Drawdown': np.nan
        }
    
    daily_returns_individual = portfolio_prices.pct_change().dropna()
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

def run_pipeline(num_stocks=300, alpha=0.01, num_random_portfolios=1000):
    """Executes the entire rolling window pipeline."""
    random.seed(42)
    
    # 1. Define analysis period and fetch all S&P 500 tickers
    all_quarters = pd.period_range(start='2020Q1', end='2025Q3', freq='Q')
    full_start_date, _ = get_quarter_dates(all_quarters[0])
    _, full_end_date = get_quarter_dates(all_quarters[-1])
    
    print("Fetching all S&P 500 tickers...")
    try:
        all_sectors, all_tickers = fetch_sp500_tickers(num_stocks=500) # Fetch all ~500
    except Exception as e:
        print(f"Fatal: Error fetching initial S&P 500 ticker list: {e}")
        return

    # 2. Load raw data for all tickers over the entire period
    print(f"Loading data for {len(all_tickers)} tickers from {full_start_date} to {full_end_date}...")
    full_raw_data = load_raw_stock_data(all_tickers, full_start_date, full_end_date)
    
    if full_raw_data.empty:
        print("Fatal: No data loaded for the entire analysis period. Exiting.")
        return
        
    # 3. Filter for a stable set of tickers with complete data
    full_close_prices = full_raw_data.xs('Close', level=1, axis=1)
    stable_tickers = full_close_prices.dropna(axis=1).columns.tolist()
    
    print(f"Found {len(stable_tickers)} tickers with complete data for the entire period.")

    # 4. Sample down to the desired number of stocks
    if len(stable_tickers) < num_stocks:
        print(f"Warning: Only {len(stable_tickers)} stable tickers found, which is less than the desired {num_stocks}. Using all stable tickers.")
        valid_tickers = stable_tickers
    else:
        print(f"Sampling {num_stocks} tickers from the stable list of {len(stable_tickers)}.")
        valid_tickers = random.sample(stable_tickers, num_stocks)

    # Pre-slice the data for the final list of valid tickers
    master_price_data = full_close_prices[valid_tickers]
    master_returns_data = master_price_data.pct_change().iloc[1:]

    # Load market data for the entire period once
    print("Loading market index data (^GSPC) for the entire period...")
    mkt_idx_all = load_market_data(full_start_date, full_end_date)

    # --- Start Rolling Window Analysis ---
    num_test_sets = len(all_quarters) - 1
    for i in range(num_test_sets):
        network_quarter = all_quarters[i]
        test_quarter = all_quarters[i+1]
        
        folder_name = f"Test_{i+1:02d}_({network_quarter}-{test_quarter})"
        output_dir = os.path.join(os.path.dirname(__file__), 'tests', folder_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n--- Processing: {folder_name} ---")

        # 5. Slice data for the current window (no new downloads)
        network_start_date, network_end_date = get_quarter_dates(network_quarter)
        test_start_date, test_end_date = get_quarter_dates(test_quarter)
        
        network_returns = master_returns_data[network_start_date:network_end_date]
        test_prices = master_price_data[test_start_date:test_end_date]
        
        network_mkt = mkt_idx_all[(mkt_idx_all['date'] >= network_start_date) & (mkt_idx_all['date'] <= network_end_date)]

        # Reshape the wide-format returns into the long format expected by the calculator
        network_returns_long = network_returns.stack().reset_index()
        network_returns_long.columns = ['date', 'ticker', 'Daily_Return']

        print("  Calculating residuals for network construction...")
        corr_matrix, corr_stats = calculate_residual_correlation(network_returns_long, network_mkt)
        
        if corr_matrix.empty:
            print("  Warning: Correlation matrix is empty. Skipping artifact generation for this period.")
            continue

        corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))

        print("  Filtering edges...")
        p_edges = threshold(corr_stats, alpha=alpha, c_min=0.0)
        G_for_centrality = create_network_from_edges(p_edges, weight_col='Correlation')
        G_for_centrality.add_nodes_from(valid_tickers)

        if p_edges.empty:
            print("  Warning: No significant edges found.")
            partition = {node: i for i, node in enumerate(valid_tickers)}
        else:
            print(f"  Found {len(p_edges)} significant edges.")
            positive_edges = p_edges[p_edges['Correlation'] > 0]
            if positive_edges.empty:
                print("  Warning: No positive edges for community detection.")
                partition = {node: i for i, node in enumerate(valid_tickers)}
            else:
                G_community = create_network_from_edges(positive_edges, weight_col='Correlation')
                G_community.add_nodes_from(valid_tickers)
                partition = detect_communities(G_community, weight_col='Correlation', random_state=42)
                print(f"  Detected {len(set(partition.values()))} communities.")
        
        inter_community_matrix = calculate_inter_community_correlation(p_edges, partition)
        inter_community_matrix.to_csv(os.path.join(output_dir, 'inter_community_correlation.csv'))

        print("  Constructing centrality portfolios...")
        centrality = calculate_centrality(G_for_centrality)
        min_centrality_portfolio, max_centrality_portfolio = [], []
        
        communities_to_tickers = {comm_id: [t for t, c in partition.items() if c == comm_id and t in valid_tickers] for comm_id in set(partition.values())}
        communities_to_tickers = {k: v for k, v in communities_to_tickers.items() if v}
        
        for comm_id, tickers_in_comm in communities_to_tickers.items():
            filtered_centrality = {t: centrality.get(t, 0) for t in tickers_in_comm}
            if not filtered_centrality: continue
            min_centrality_portfolio.append(min(filtered_centrality, key=filtered_centrality.get))
            max_centrality_portfolio.append(max(filtered_centrality, key=filtered_centrality.get))

        # Save the composition of the main strategy portfolios
        portfolios_to_save = {
            'min_centrality_portfolio': min_centrality_portfolio,
            'max_centrality_portfolio': max_centrality_portfolio
        }
        with open(os.path.join(output_dir, 'portfolios.json'), 'w') as f: json.dump(portfolios_to_save, f, indent=4)

        print("  Calculating performance for main and random portfolios...")
        all_performance_results = []

        # Performance of main strategies
        perf_min = get_portfolio_performance(test_prices[min_centrality_portfolio] if min_centrality_portfolio else pd.DataFrame())
        perf_min['portfolio_type'] = 'min_centrality'
        all_performance_results.append(perf_min)

        perf_max = get_portfolio_performance(test_prices[max_centrality_portfolio] if max_centrality_portfolio else pd.DataFrame())
        perf_max['portfolio_type'] = 'max_centrality'
        all_performance_results.append(perf_max)
        
        # Performance of random portfolios
        num_to_select = len(communities_to_tickers)
        if num_to_select > 0 and len(valid_tickers) >= num_to_select:
            for i in range(num_random_portfolios):
                random_tickers = random.sample(valid_tickers, num_to_select)
                perf_random = get_portfolio_performance(test_prices[random_tickers])
                perf_random['portfolio_type'] = 'random'
                all_performance_results.append(perf_random)
        
        # Save all performance results to a single CSV
        performance_df = pd.DataFrame(all_performance_results)
        performance_df.to_csv(os.path.join(output_dir, 'backtest_results.csv'), index=False)
        print(f"  Saved performance for 2 main and {num_random_portfolios} random portfolios.")

        G_final_visual = create_network_from_edges(p_edges, weight_col='Correlation', edge_attrs=['Correlation'])
        G_final_visual.add_nodes_from(valid_tickers)
        viz_path = os.path.join(output_dir, 'network_visualization.png')
        print("  Generating and saving network visualization...")
        visualize_network(G_final_visual, partition, output_filename=viz_path)
        print(f"--- Completed: {folder_name} ---")

if __name__ == '__main__':
    run_pipeline(num_stocks=300)