# 주식 네트워크 기반 포트폴리오 백테스팅 파이프라인 (Refactored)
# 전체 프로세스: 데이터 로드 -> 분기별 분석(네트워크 구축, 커뮤니티 탐지, 백테스팅) -> 결과 저장

import os
import pandas as pd
import numpy as np
import random
import json
import networkx as nx
import community as community_louvain
import shutil
import matplotlib.pyplot as plt

import config
from pipeline.data_loader import fetch_sp500_tickers, load_raw_stock_data, load_market_data
from pipeline.corr_calculator import calculate_residual_correlation
from pipeline.threshold import threshold
from pipeline.utils import get_quarter_dates
from pipeline.portfolio import get_portfolio_performance
from pipeline.network_analysis import (
    calculate_centrality,
    create_network_from_edges,
    detect_communities,
    visualize_network,
    visualize_degree_ccdf,
    visualize_degree_histogram,
    visualize_degree_pdf
)
from pipeline.weighted_portfolio import select_weighted_portfolio

def setup_directories():
    """결과물 저장을 위한 디렉토리 구조 생성 및 초기화"""
    if os.path.exists(config.BASE_OUTPUT_DIR):
        shutil.rmtree(config.BASE_OUTPUT_DIR)
    
    os.makedirs(config.SUMMARY_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.QUARTERLY_DIR, exist_ok=True)

def load_data():
    """분석에 필요한 데이터 로드"""
    print("Fetching S&P 500 tickers...")
    try:
        sectors_df, all_tickers = fetch_sp500_tickers()
        sector_map = dict(zip(sectors_df["ticker"], sectors_df["GICS Sector"]))
    except Exception as e:
        print(f"Fatal Error: {e}")
        return None

    print(f"Loading prices for {len(all_tickers)} tickers...")
    raw_data = load_raw_stock_data(all_tickers, config.START_DATE, config.END_DATE)
    if raw_data.empty: return None
    
    prices = raw_data.xs('Close', level=1, axis=1)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    
    print("Loading market index...")
    market_data = load_market_data(config.START_DATE, config.END_DATE)
    
    return prices.columns.tolist(), prices, returns, market_data, sector_map

def run_quarter_analysis(i, quarter, valid_tickers, prices, returns, market_data, sector_map):
    """단일 분기 분석 수행"""
    q_str = str(quarter)
    q_dir = os.path.join(config.QUARTERLY_DIR, q_str)
    
    dirs = {
        'root': q_dir,
        'network': os.path.join(q_dir, 'network_data'),
        'clusters': os.path.join(q_dir, 'clusters'),
        'portfolios': os.path.join(q_dir, 'portfolios'),
        'figures': os.path.join(q_dir, 'figures')
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)
    
    print(f"\n--- Processing: {q_str} ---")

    start_date, end_date = get_quarter_dates(quarter)
    q_returns = returns[start_date:end_date]
    q_prices = prices[start_date:end_date]
    q_market = market_data[(market_data['date'] >= start_date) & (market_data['date'] <= end_date)]

    has_data = q_returns.notna().any() & q_prices.notna().any()
    current_tickers = [t for t in valid_tickers if has_data.get(t, False)]
    
    if not current_tickers:
        print("  Warning: No valid tickers.")
        return None

    # 네트워크 구축
    q_returns_long = q_returns.stack(future_stack=True).reset_index()
    q_returns_long.columns = ['date', 'ticker', 'Daily_Return']
    
    corr_matrix, corr_stats = calculate_residual_correlation(q_returns_long, q_market)
    if not corr_matrix.empty:
        corr_matrix.to_csv(os.path.join(dirs['network'], 'correlation_matrix.csv'))

    p_edges = threshold(corr_stats, alpha=config.ALPHA, c_min=config.CORRELATION_THRESHOLD)
    
    if not p_edges.empty:
        gephi_edges = p_edges[['ticker1', 'ticker2', 'Correlation']].rename(
            columns={'ticker1': 'source', 'ticker2': 'target', 'Correlation': 'weight'}
        )
        gephi_edges.to_csv(os.path.join(dirs['network'], 'gephi_edges.csv'), index=False)

    # 커뮤니티 탐지
    modularity = 0.0
    partition = {}
    
    if not p_edges.empty:
        pos_edges = p_edges[p_edges['Correlation'] > 0]
        if not pos_edges.empty:
            G_comm = create_network_from_edges(pos_edges)
            G_comm.add_nodes_from(current_tickers)
            partition = detect_communities(G_comm, random_state=42)
            try:
                modularity = community_louvain.modularity(partition, G_comm, weight='Correlation')
            except: pass
        else:
            partition = {n: i for i, n in enumerate(current_tickers)}
    else:
        partition = {n: i for i, n in enumerate(current_tickers)}

    print(f"  Detected {len(set(partition.values()))} communities.")

    # 클러스터 정보 저장
    if partition:
        nodes_df = pd.DataFrame(list(partition.items()), columns=['Ticker', 'Cluster_ID'])
        nodes_df['Sector'] = nodes_df['Ticker'].map(sector_map).fillna('Unknown')
        nodes_df.sort_values(['Cluster_ID', 'Ticker']).to_csv(os.path.join(dirs['clusters'], 'nodes.csv'), index=False)
    
    if not p_edges.empty and partition:
        edges_df = p_edges.copy()
        edges_df['Source_Cluster'] = edges_df['ticker1'].map(partition)
        edges_df['Target_Cluster'] = edges_df['ticker2'].map(partition)
        edges_df['Edge_Type'] = np.where(
            edges_df['Source_Cluster'] == edges_df['Target_Cluster'], 'Internal', 'External'
        )
        edges_df.to_csv(os.path.join(dirs['clusters'], 'edges.csv'), index=False)

    # 포트폴리오 구성
    min_pf, max_pf = [], []
    comm_groups = {}
    for t, c in partition.items():
        if t in current_tickers: comm_groups.setdefault(c, []).append(t)
        
    for cid, members in comm_groups.items():
        if len(members) < 2:
            min_pf.extend(members)
            max_pf.extend(members)
            continue
            
        sub_edges = p_edges[
            (p_edges['ticker1'].isin(members)) & (p_edges['ticker2'].isin(members)) &
            (p_edges['Correlation'] > 0)
        ]
        
        if sub_edges.empty: continue
        
        G_sub = create_network_from_edges(sub_edges)
        G_sub.add_nodes_from(members)
        cent = calculate_centrality(G_sub)
        
        if cent:
            sorted_m = sorted(cent.keys(), key=lambda x: (cent.get(x, 0), x))
            min_pf.append(sorted_m[0])
            max_pf.append(sorted_m[-1])

    weighted_pf = select_weighted_portfolio(q_prices, sector_map, random_state=42 + i)
    
    portfolios = {
        'min_eigenvector': min_pf,
        'max_eigenvector': max_pf,
        'weighted_sector': weighted_pf
    }
    with open(os.path.join(dirs['portfolios'], 'portfolios.json'), 'w') as f:
        json.dump(portfolios, f, indent=4)

    # 백테스팅
    results = []
    for name, tickers in portfolios.items():
        res = get_portfolio_performance(q_prices[tickers] if tickers else pd.DataFrame())
        res['portfolio_type'] = name
        results.append(res)
        
    n_select = len(min_pf)
    if n_select > 0 and len(current_tickers) >= n_select:
        for _ in range(config.NUM_RANDOM_PORTFOLIOS):
            rnd_tickers = random.sample(current_tickers, n_select)
            res = get_portfolio_performance(q_prices[rnd_tickers])
            res['portfolio_type'] = 'random'
            results.append(res)
            
    perf_df = pd.DataFrame(results)
    perf_df.to_csv(os.path.join(dirs['portfolios'], 'backtest_results.csv'), index=False)

    # 시각화
    G_vis = create_network_from_edges(p_edges)
    G_vis.add_nodes_from(current_tickers)
    visualize_network(G_vis, partition, os.path.join(dirs['figures'], 'network_viz.png'))
    
    base_fig = os.path.join(dirs['figures'], 'degree_dist')
    visualize_degree_ccdf(G_vis, f"{base_fig}_ccdf.png")
    visualize_degree_histogram(G_vis, f"{base_fig}_hist.png")
    visualize_degree_pdf(G_vis, f"{base_fig}_pdf.png")

    num_nodes = G_vis.number_of_nodes()
    num_edges = G_vis.number_of_edges()
    avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
    density = 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    
    return {
        'Quarter': str(quarter),
        'Num_Nodes': num_nodes,
        'Num_Edges': num_edges,
        'Num_Communities': len(set(partition.values())),
        'Modularity': modularity,
        'avg_degree': avg_degree,
        'network_density': density
    }

def main():
    random.seed(42)
    setup_directories()
    
    data = load_data()
    if not data: return
    
    quarters = pd.period_range(start=config.START_QUARTER, end=config.END_QUARTER, freq='Q')
    metrics_list = []
    
    for i, q in enumerate(quarters):
        metrics = run_quarter_analysis(i, q, *data)
        if metrics: metrics_list.append(metrics)
        
    if metrics_list:
        pd.DataFrame(metrics_list).to_csv(os.path.join(config.SUMMARY_DIR, 'network_metrics.csv'), index=False)
        print("\nPipeline Completed Successfully.")

if __name__ == '__main__':
    main()