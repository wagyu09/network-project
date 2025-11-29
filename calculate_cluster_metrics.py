
import os
import pandas as pd
import numpy as np
import community as community_louvain
import networkx as nx
import config
from pipeline.data_loader import fetch_sp500_tickers, load_raw_stock_data, load_market_data
from pipeline.corr_calculator import calculate_residual_correlation
from pipeline.threshold import threshold
from pipeline.network_analysis import create_network_from_edges, detect_communities
from pipeline.utils import get_quarter_dates

def load_data():
    print("Fetching all S&P 500 tickers...")
    try:
        sectors_df, all_tickers = fetch_sp500_tickers()
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return None, None, None, None

    print(f"Loading data for {len(all_tickers)} tickers from {config.START_DATE} to {config.END_DATE}...")
    full_raw_data = load_raw_stock_data(all_tickers, config.START_DATE, config.END_DATE)
    
    if full_raw_data.empty:
        return None, None, None, None
        
    full_close_prices = full_raw_data.xs('Close', level=1, axis=1)
    valid_tickers = full_close_prices.columns.tolist()
    master_returns_data = full_close_prices.pct_change(fill_method=None).iloc[1:]
    
    print("Loading market index data...")
    mkt_idx_all = load_market_data(config.START_DATE, config.END_DATE)

    return valid_tickers, master_returns_data, mkt_idx_all

def calculate_metrics():
    valid_tickers, master_returns_data, mkt_idx_all = load_data()
    if valid_tickers is None:
        return

    all_quarters = pd.period_range(start=config.START_QUARTER, end=config.END_QUARTER, freq='Q')
    
    results = []
    
    print("\nStarting Quarterly Analysis...")
    for q in all_quarters:
        print(f"Processing {q}...")
        start_date, end_date = get_quarter_dates(q)
        
        network_returns = master_returns_data[start_date:end_date]
        network_mkt = mkt_idx_all[(mkt_idx_all['date'] >= start_date) & (mkt_idx_all['date'] <= end_date)]
        
        # Data availability check
        has_data = network_returns.notna().any()
        current_tickers = [t for t in valid_tickers if has_data.get(t, False)]
        
        if len(current_tickers) < 2:
            continue
            
        network_returns_long = network_returns[current_tickers].stack(future_stack=True).reset_index()
        network_returns_long.columns = ['date', 'ticker', 'Daily_Return']
        
        # Calculate Correlations
        corr_matrix, corr_stats = calculate_residual_correlation(network_returns_long, network_mkt)
        if corr_matrix.empty:
            continue
            
        # Filter Edges for Community Detection
        p_edges = threshold(corr_stats, alpha=config.ALPHA, c_min=config.CORRELATION_THRESHOLD)
        
        # Detect Communities
        partition = {}
        if not p_edges.empty:
            positive_edges = p_edges[p_edges['Correlation'] > 0]
            if not positive_edges.empty:
                G = create_network_from_edges(positive_edges, weight_col='Correlation')
                # Ensure all current tickers are in the graph (even if isolated)
                G.add_nodes_from(current_tickers)
                partition = detect_communities(G, weight_col='Correlation', random_state=42)
            else:
                partition = {t: 0 for t in current_tickers}
        else:
            partition = {t: 0 for t in current_tickers}
            
        # Prepare Absolute Correlation Matrix for Metrics
        abs_corr_matrix = corr_matrix.abs()
        
        # Group tickers by community
        comm_to_tickers = {}
        for node, comm_id in partition.items():
            if node in current_tickers: # valid tickers only
                comm_to_tickers.setdefault(comm_id, []).append(node)
                
        sorted_comm_ids = sorted(comm_to_tickers.keys())
        
        # Calculate Metrics
        for comm_id in sorted_comm_ids:
            tickers = comm_to_tickers[comm_id]
            n = len(tickers)
            
            # Internal Cohesion
            if n > 1:
                sub_matrix = abs_corr_matrix.loc[tickers, tickers]
                sum_corr = sub_matrix.values.sum()
                # Subtract diagonal (1s)
                sum_corr -= n
                internal_cohesion = sum_corr / (n * (n - 1))
            else:
                internal_cohesion = 1.0 # Self-cohesion is 1
                
            # Inter-cluster Correlations
            inter_corrs = []
            for other_id in sorted_comm_ids:
                if comm_id == other_id:
                    continue
                
                other_tickers = comm_to_tickers[other_id]
                if not other_tickers:
                    continue
                    
                sub_matrix_inter = abs_corr_matrix.loc[tickers, other_tickers]
                avg_inter = sub_matrix_inter.values.mean()
                inter_corrs.append(avg_inter)
            
            avg_external_corr = np.mean(inter_corrs) if inter_corrs else 0.0
            
            results.append({
                'Quarter': str(q),
                'Cluster_ID': comm_id,
                'Size': n,
                'Internal_Cohesion': internal_cohesion,
                'Avg_External_Correlation': avg_external_corr
            })

    # Save Results
    df_results = pd.DataFrame(results)
    df_results.to_csv('cluster_metrics_analysis.csv', index=False)
    print("\nAnalysis Complete. Results saved to 'cluster_metrics_analysis.csv'.")
    print("\n--- Summary of First 10 Rows ---")
    print(df_results.head(10))

    # Calculate and print Period Averages
    print("\n--- Quarterly Averages ---")
    quarterly_avg = df_results.groupby('Quarter')[['Internal_Cohesion', 'Avg_External_Correlation']].mean()
    print(quarterly_avg)
    quarterly_avg.to_csv('quarterly_cluster_averages.csv')

if __name__ == '__main__':
    calculate_metrics()
