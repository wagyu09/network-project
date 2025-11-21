import os
import sys
import pandas as pd
import numpy as np
import random
import json
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from data_loader import fetch_sp500_tickers, load_raw_stock_data, load_market_data
from corr_calculator import get_residuals, calculate_residual_correlation
from threshold import threshold
from utils import get_quarter_dates
from portfolio import get_portfolio_performance
from network_analysis import (
    calculate_centrality,
    create_network_from_edges,
    detect_communities,
    calculate_inter_community_correlation,
    visualize_network
)

def load_and_prepare_data():
    """
    Loads all necessary data and prepares master dataframes for the analysis.
    """
    print("Fetching all S&P 500 tickers...")
    try:
        _, all_tickers = fetch_sp500_tickers()
    except Exception as e:
        print(f"Fatal: Error fetching initial S&P 500 ticker list: {e}")
        return None, None, None, None

    # Load raw data for all tickers over the entire period
    print(f"Loading data for {len(all_tickers)} tickers from {config.START_DATE} to {config.END_DATE}...")
    full_raw_data = load_raw_stock_data(all_tickers, config.START_DATE, config.END_DATE)
    
    if full_raw_data.empty:
        print("Fatal: No data loaded for the entire analysis period. Exiting.")
        return None, None, None, None
        
    # Use all tickers for which data was successfully downloaded
    full_close_prices = full_raw_data.xs('Close', level=1, axis=1)
    valid_tickers = full_close_prices.columns.tolist()
    print(f"Using {len(valid_tickers)} tickers for which data was successfully downloaded.")

    master_price_data = full_close_prices
    master_returns_data = master_price_data.pct_change(fill_method=None).iloc[1:]

    print("Loading market index data (^GSPC) for the entire period...")
    mkt_idx_all = load_market_data(config.START_DATE, config.END_DATE)

    return valid_tickers, master_price_data, master_returns_data, mkt_idx_all

def run_single_quarter_analysis(i, network_quarter, test_quarter, valid_tickers, master_price_data, master_returns_data, mkt_idx_all, alpha, num_random_portfolios):
    """
    Runs the entire analysis pipeline for a single quarter.
    """
    folder_name = f"Test_{i+1:02d}_({network_quarter}-{test_quarter})"
    output_dir = os.path.join(config.TESTS_OUTPUT_DIR, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Processing: {folder_name} ---")

    network_start_date, network_end_date = get_quarter_dates(network_quarter)
    test_start_date, test_end_date = get_quarter_dates(test_quarter)
    
    network_returns = master_returns_data[network_start_date:network_end_date]
    test_prices = master_price_data[test_start_date:test_end_date]
    
    network_mkt = mkt_idx_all[(mkt_idx_all['date'] >= network_start_date) & (mkt_idx_all['date'] <= network_end_date)]

    network_returns_long = network_returns.stack(future_stack=True).reset_index()
    network_returns_long.columns = ['date', 'ticker', 'Daily_Return']

    print("  Calculating residuals for network construction...")
    corr_matrix, corr_stats = calculate_residual_correlation(network_returns_long, network_mkt)
    
    if corr_matrix.empty:
        print("  Warning: Correlation matrix is empty. Skipping artifact generation for this period.")
        return

    corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))

    print("  Filtering edges...")
    p_edges = threshold(corr_stats, alpha=alpha, c_min=0.0)
    G_for_centrality = create_network_from_edges(p_edges, weight_col='Correlation')
    G_for_centrality.add_nodes_from(valid_tickers)

    # --- ADDED: Save edge list for Gephi ---
    print("  Saving edge list for Gephi...")
    gephi_edges = p_edges[['ticker1', 'ticker2', 'Correlation']].copy()
    gephi_edges.rename(columns={'ticker1': 'source', 'ticker2': 'target', 'Correlation': 'weight'}, inplace=True)
    gephi_path = os.path.join(output_dir, 'gephi_edges.csv')
    gephi_edges.to_csv(gephi_path, index=False)
    # --- END ADDED ---

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

    portfolios_to_save = {'min_centrality_portfolio': min_centrality_portfolio, 'max_centrality_portfolio': max_centrality_portfolio}
    with open(os.path.join(output_dir, 'portfolios.json'), 'w') as f: json.dump(portfolios_to_save, f, indent=4)

    print("  Calculating performance for main and random portfolios...")
    all_performance_results = []

    perf_min = get_portfolio_performance(test_prices[min_centrality_portfolio] if min_centrality_portfolio else pd.DataFrame())
    perf_min['portfolio_type'] = 'min_centrality'
    all_performance_results.append(perf_min)

    perf_max = get_portfolio_performance(test_prices[max_centrality_portfolio] if max_centrality_portfolio else pd.DataFrame())
    perf_max['portfolio_type'] = 'max_centrality'
    all_performance_results.append(perf_max)
    
    num_to_select = len(communities_to_tickers)
    if num_to_select > 0 and len(valid_tickers) >= num_to_select:
        for i in range(num_random_portfolios):
            random_tickers = random.sample(valid_tickers, num_to_select)
            perf_random = get_portfolio_performance(test_prices[random_tickers])
            perf_random['portfolio_type'] = 'random'
            all_performance_results.append(perf_random)
    
    performance_df = pd.DataFrame(all_performance_results)
    performance_df.to_csv(os.path.join(output_dir, 'backtest_results.csv'), index=False)
    print(f"  Saved performance for 2 main and {num_random_portfolios} random portfolios.")

    G_final_visual = create_network_from_edges(p_edges, weight_col='Correlation', edge_attrs=['Correlation'])
    G_final_visual.add_nodes_from(valid_tickers)
    viz_path = os.path.join(output_dir, 'network_visualization.png')
    print("  Generating and saving network visualization...")
    visualize_network(G_final_visual, partition, output_filename=viz_path)
    print(f"--- Completed: {folder_name} ---")

def run_pipeline(alpha=config.ALPHA, num_random_portfolios=config.NUM_RANDOM_PORTFOLIOS):
    """
    Orchestrates the entire rolling window analysis.
    """
    random.seed(42)
    
    # 1. Load and prepare all data for the entire period
    valid_tickers, master_price_data, master_returns_data, mkt_idx_all = load_and_prepare_data()
    if valid_tickers is None:
        return

    # 2. Run rolling window analysis
    all_quarters = pd.period_range(start=config.START_QUARTER, end=config.END_QUARTER, freq='Q')
    for i in range(len(all_quarters) - 1):
        network_quarter = all_quarters[i]
        test_quarter = all_quarters[i+1]
        run_single_quarter_analysis(
            i, network_quarter, test_quarter,
            valid_tickers, master_price_data, master_returns_data, mkt_idx_all,
            alpha, num_random_portfolios
        )

if __name__ == '__main__':
    run_pipeline()