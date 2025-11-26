"""주식 네트워크 기반 포트폴리오 백테스팅 파이프라인의 메인 실행 스크립트

시장 효과가 제거된 주식 상관관계 네트워크를 구축하고, 커뮤니티 탐지 및
중심성 분석을 통해 포트폴리오를 구성함. 구성된 포트폴리오의 성과를
무작위 포트폴리오와 비교하여 전략의 유효성을 검증하는 것을 목적으로 함
"""
import os
import pandas as pd
import numpy as np
import random
import json
import networkx as nx
import community as community_louvain # 모듈러리티 계산을 위해 추가

import config
from pipeline.data_loader import fetch_sp500_tickers, load_raw_stock_data, load_market_data
from pipeline.corr_calculator import get_residuals, calculate_residual_correlation
from pipeline.threshold import threshold
from pipeline.utils import get_quarter_dates
from pipeline.portfolio import get_portfolio_performance
from pipeline.network_analysis import (
    calculate_centrality,
    create_network_from_edges,
    detect_communities,
    visualize_network,
    visualize_degree_distribution
)

def load_and_prepare_data() -> tuple[list, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """분석에 필요한 모든 데이터를 로드하고 마스터 데이터프레임을 준비

    S&P 500 티커 목록, 전체 기간의 주가 및 수익률 데이터, 시장 지수 데이터를 로드
    이 단계에서 데이터를 미리 로드하여 각 분기 분석 시 디스크 I/O를 최소화함

    Returns:
        valid_tickers (list): 유효한 티커 목록
        master_price_data (pd.DataFrame): 전체 기간의 마스터 종가 데이터
        master_returns_data (pd.DataFrame): 전체 기간의 마스터 수익률 데이터
        mkt_idx_all (pd.DataFrame): 전체 기간의 시장 지수 데이터
    """
    print("Fetching all S&P 500 tickers...")
    try:
        _, all_tickers = fetch_sp500_tickers()
    except Exception as e:
        print(f"Fatal: Error fetching initial S&P 500 ticker list: {e}")
        return None, None, None, None

    # 전체 기간에 대한 원시 데이터 로드
    print(f"Loading data for {len(all_tickers)} tickers from {config.START_DATE} to {config.END_DATE}...")
    full_raw_data = load_raw_stock_data(all_tickers, config.START_DATE, config.END_DATE)
    
    if full_raw_data.empty:
        print("Fatal: No data loaded for the entire analysis period. Exiting.")
        return None, None, None, None
        
    # 다운로드된 유효 티커 목록 추출
    full_close_prices = full_raw_data.xs('Close', level=1, axis=1)
    valid_tickers = full_close_prices.columns.tolist()
    print(f"Using {len(valid_tickers)} tickers for which data was successfully downloaded.")

    master_price_data = full_close_prices
    # 마스터 수익률 데이터 계산 (첫 행은 NaN이므로 제외)
    master_returns_data = master_price_data.pct_change(fill_method=None).iloc[1:]

    # 전체 기간에 대한 시장 지수 데이터 로드
    print("Loading market index data (^GSPC) for the entire period...")
    mkt_idx_all = load_market_data(config.START_DATE, config.END_DATE)

    return valid_tickers, master_price_data, master_returns_data, mkt_idx_all

def run_single_quarter_analysis(i: int, network_quarter: pd.Period, test_quarter: pd.Period,
                                valid_tickers: list, master_price_data: pd.DataFrame,
                                master_returns_data: pd.DataFrame, mkt_idx_all: pd.DataFrame,
                                alpha: float, num_random_portfolios: int):
    """단일 분기에 대한 네트워크 분석 및 백테스팅을 실행

    네트워크 구축 -> 군집 탐지 -> 포트폴리오 구성 -> 성과 계산의 전체 과정을
    특정 분기(quarter)에 대해 수행하고 결과를 파일로 저장함

    Args:
        i (int): 현재 분기 인덱스 (폴더명 생성에 사용)
        network_quarter (pd.Period): 네트워크 구축용 분기
        test_quarter (pd.Period): 포트폴리오 성과 측정(백테스팅)용 분기
        (이하 Args): load_and_prepare_data에서 로드된 마스터 데이터 및 설정값
    """
    # 1. 경로 설정 및 데이터 슬라이싱
    folder_name = f"Test_{i+1:02d}_({network_quarter}-{test_quarter})"
    output_dir = os.path.join(config.TESTS_OUTPUT_DIR, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Processing: {folder_name} ---")

    # 현재 분기에 해당하는 데이터만 마스터 데이터프레임에서 추출
    network_start_date, network_end_date = get_quarter_dates(network_quarter)
    test_start_date, test_end_date = get_quarter_dates(test_quarter)
    
    network_returns = master_returns_data[network_start_date:network_end_date]
    test_prices = master_price_data[test_start_date:test_end_date]
    network_mkt = mkt_idx_all[(mkt_idx_all['date'] >= network_start_date) & (mkt_idx_all['date'] <= network_end_date)]

    # --- Data Availability Check ---
    # Filter tickers to ensure they have valid data in both the network construction AND test periods.
    # This prevents "future" stocks (like spin-offs) from being selected in past periods where they had no data,
    # which causes portfolio calculations to return empty/NaN results.
    has_data_network = network_returns.notna().any()
    has_data_test = test_prices.notna().any()
    
    valid_tickers = [
        t for t in valid_tickers 
        if has_data_network.get(t, False) and has_data_test.get(t, False)
    ]
    
    if not valid_tickers:
        print("  Warning: No valid tickers found with data for this period. Skipping.")
        return

    print(f"  Active tickers for this period: {len(valid_tickers)}")

    # 2. 네트워크 구축
    # 2-1. 시장 효과 제거 (잔차 계산) 및 상관관계 행렬 생성
    network_returns_long = network_returns.stack(future_stack=True).reset_index()
    network_returns_long.columns = ['date', 'ticker', 'Daily_Return']

    print("  Calculating residuals for network construction...")
    corr_matrix, corr_stats = calculate_residual_correlation(network_returns_long, network_mkt)
    
    if corr_matrix.empty:
        print("  Warning: Correlation matrix is empty. Skipping artifact generation for this period.")
        return

    corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))

    # 2-2. 통계적 유의성에 기반한 엣지 필터링
    print(f"  Filtering edges (alpha={alpha}, c_min={config.CORRELATION_THRESHOLD})...")
    p_edges = threshold(corr_stats, alpha=alpha, c_min=config.CORRELATION_THRESHOLD)
    
    # --- Gephi 시각화용 엣지 리스트 저장 ---
    gephi_edges = p_edges[['ticker1', 'ticker2', 'Correlation']].copy()
    gephi_edges.rename(columns={'ticker1': 'source', 'ticker2': 'target', 'Correlation': 'weight'}, inplace=True)
    gephi_path = os.path.join(output_dir, 'gephi_edges.csv')
    gephi_edges.to_csv(gephi_path, index=False)

    # 3. 네트워크 분석 및 포트폴리오 구성
    # 3-1. 커뮤니티 탐지 (Louvain 알고리즘)
    modularity_score = 0.0 # 초기화
    
    if p_edges.empty:
        print("  Warning: No significant edges found.")
        partition = {node: i for i, node in enumerate(valid_tickers)}
    else:
        print(f"  Found {len(p_edges)} significant edges.")
        # 양의 상관관계만으로 커뮤니티를 형성해야 그룹의 의미가 명확해짐
        positive_edges = p_edges[p_edges['Correlation'] > 0]
        if positive_edges.empty:
            print("  Warning: No positive edges for community detection.")
            partition = {node: i for i, node in enumerate(valid_tickers)}
        else:
            G_community = create_network_from_edges(positive_edges, weight_col='Correlation')
            G_community.add_nodes_from(valid_tickers) # 모든 티커를 노드로 포함
            partition = detect_communities(G_community, weight_col='Correlation', random_state=42)
            print(f"  Detected {len(set(partition.values()))} communities.")
            
            # 모듈러리티 점수 계산
            try:
                modularity_score = community_louvain.modularity(partition, G_community, weight='Correlation')
            except ValueError:
                modularity_score = 0.0
    
    # 3-2. 포트폴리오 구성 (지역 중심성 기반)
    print("  Constructing centrality portfolios (Eigenvector)...")
    min_eigenvector_portfolio, max_eigenvector_portfolio = [], []
    
    # 각 커뮤니티 ID에 해당하는 티커들을 그룹화
    communities_to_tickers = {comm_id: [t for t, c in partition.items() if c == comm_id and t in valid_tickers] for comm_id in set(partition.values())}
    communities_to_tickers = {k: v for k, v in communities_to_tickers.items() if v}
    
    # 각 커뮤니티(군집)별로 순회하며 포트폴리오 종목 선택
    for comm_id, tickers_in_comm in communities_to_tickers.items():
        # 해당 커뮤니티에 속한 엣지만으로 '서브그래프'를 생성
        community_edges = p_edges[
            (p_edges['ticker1'].isin(tickers_in_comm)) & 
            (p_edges['ticker2'].isin(tickers_in_comm))
        ]
        
        # 커뮤니티 내에 엣지가 있고, 종목이 2개 이상일 때만 중심성 계산
        if not community_edges.empty and len(tickers_in_comm) > 1:
            # [수정됨] 중심성 계산 시 양의 상관관계(Correlation > 0)만 사용
            positive_community_edges = community_edges[community_edges['Correlation'] > 0]
            
            if positive_community_edges.empty:
                 continue

            G_community_subgraph = create_network_from_edges(positive_community_edges, weight_col='Correlation')
            G_community_subgraph.add_nodes_from(tickers_in_comm)
            
            # 해당 서브그래프 내에서 '지역 중심성(Eigenvector)' 계산
            local_centrality = calculate_centrality(G_community_subgraph)
            
            # 중심성 점수가 가장 낮은 종목과 높은 종목을 선택 (각 군집당 1개 고정)
            if local_centrality:
                # 재현성 보장을 위한 2차 정렬 기준 추가:
                # 1. 중심성 점수 기준 정렬
                # 2. (동점일 경우) 티커 이름 알파벳순으로 정렬
                sorted_tickers = sorted(local_centrality.keys(), key=lambda t: (local_centrality.get(t, 0), t))
                
                # 각 군집당 1개씩만 선정 (대표 선수)
                n_select = 1
                
                # 하위 1개 선정
                min_tickers = sorted_tickers[:n_select]
                min_eigenvector_portfolio.extend(min_tickers)
                
                # 상위 1개 선정
                max_tickers = sorted_tickers[-n_select:]
                max_eigenvector_portfolio.extend(max_tickers)
        
        # 커뮤니티에 종목이 하나만 있으면, 그 종목을 양쪽 포트폴리오에 모두 포함
        elif len(tickers_in_comm) == 1:
            min_eigenvector_portfolio.append(tickers_in_comm[0])
            max_eigenvector_portfolio.append(tickers_in_comm[0])

    # 구성된 포트폴리오 종목 리스트를 json 파일로 저장
    portfolios_to_save = {'min_eigenvector_portfolio': min_eigenvector_portfolio, 'max_eigenvector_portfolio': max_eigenvector_portfolio}
    with open(os.path.join(output_dir, 'portfolios.json'), 'w') as f: json.dump(portfolios_to_save, f, indent=4)

    # 4. 백테스팅 및 성과 분석
    print("  Calculating performance for main and random portfolios...")
    all_performance_results = []

    # 4-1. 주요 전략 포트폴리오 성과 계산
    perf_min = get_portfolio_performance(test_prices[min_eigenvector_portfolio] if min_eigenvector_portfolio else pd.DataFrame())
    perf_min['portfolio_type'] = 'min_eigenvector'
    all_performance_results.append(perf_min)

    perf_max = get_portfolio_performance(test_prices[max_eigenvector_portfolio] if max_eigenvector_portfolio else pd.DataFrame())
    perf_max['portfolio_type'] = 'max_eigenvector'
    all_performance_results.append(perf_max)
    
    # 4-2. 비교 분석을 위한 무작위 포트폴리오 성과 계산
    # [수정됨] 공정한 비교를 위해 전략 포트폴리오와 동일한 수의 종목을 무작위로 선택
    num_to_select = len(min_eigenvector_portfolio)
    
    if num_to_select > 0 and len(valid_tickers) >= num_to_select:
        for _ in range(num_random_portfolios):
            random_tickers = random.sample(valid_tickers, num_to_select)
            perf_random = get_portfolio_performance(test_prices[random_tickers])
            perf_random['portfolio_type'] = 'random'
            all_performance_results.append(perf_random)
    else:
        print(f"  Warning: Skipping random portfolios. Strategy size: {num_to_select}, Valid tickers: {len(valid_tickers)}")
    
    # 모든 성과 결과를 하나의 CSV 파일로 저장
    performance_df = pd.DataFrame(all_performance_results)
    performance_df.to_csv(os.path.join(output_dir, 'backtest_results.csv'), index=False)
    print(f"  Saved performance for 2 main and {num_random_portfolios} random portfolios.")

    # 5. 네트워크 시각화 생성 및 저장
    G_final_visual = create_network_from_edges(p_edges, weight_col='Correlation', edge_attrs=['Correlation'])
    G_final_visual.add_nodes_from(valid_tickers)
    viz_path = os.path.join(output_dir, 'network_visualization.png')
    print("  Generating and saving network visualization...")
    visualize_network(G_final_visual, partition, output_filename=viz_path)

    # 5-2. 디그리 분포 시각화 생성 및 저장
    degree_dist_filename = f"degree_distribution_{network_quarter}-{test_quarter}.png"
    degree_dist_path = os.path.join('degree_distributions', degree_dist_filename)
    visualize_degree_distribution(G_final_visual, output_filename=degree_dist_path)

    print(f"--- Completed: {folder_name} ---")
    
    # 최종 통계량 계산
    num_nodes = G_final_visual.number_of_nodes()
    num_edges = G_final_visual.number_of_edges()
    
    avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
    density = 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

    return {
        'Quarter': str(network_quarter),
        'Num_Nodes': num_nodes,
        'Num_Edges': num_edges,
        'Num_Communities': len(set(partition.values())) if partition else 0,
        'Modularity': modularity_score,
        'avg_degree': avg_degree,
        'network_density': density
    }

def run_pipeline(alpha: float = config.ALPHA, num_random_portfolios: int = config.NUM_RANDOM_PORTFOLIOS):
    """주식 네트워크 기반 포트폴리오 백테스팅 파이프라인의 전체 실행을 조율

    Args:
        alpha (float): 엣지 필터링 유의수준 (config 파일에서 로드)
        num_random_portfolios (int): 생성할 무작위 포트폴리오 개수 (config 파일에서 로드)
    """
    random.seed(42) # 결과 재현성을 위한 시드 고정
    
    # 그래프 이미지를 저장할 폴더 생성
    os.makedirs('degree_distributions', exist_ok=True)
    
    # 1. 전체 분석 기간 데이터 미리 로드
    valid_tickers, master_price_data, master_returns_data, mkt_idx_all = load_and_prepare_data()
    if valid_tickers is None:
        return

    # 2. 롤링 윈도우 방식으로 전체 기간 분석 실행
    all_quarters = pd.period_range(start=config.START_QUARTER, end=config.END_QUARTER, freq='Q')
    network_stats_list = []
    
    for i in range(len(all_quarters) - 1):
        network_quarter = all_quarters[i] # 네트워크 구축 분기
        test_quarter = all_quarters[i+1]    # 포트폴리오 성과 측정 분기
        stats = run_single_quarter_analysis(
            i, network_quarter, test_quarter,
            valid_tickers, master_price_data, master_returns_data, mkt_idx_all,
            alpha, num_random_portfolios
        )
        if stats:
            network_stats_list.append(stats)
            
    # 네트워크 통계 저장
    if network_stats_list:
        stats_df = pd.DataFrame(network_stats_list)
        stats_df.to_csv('network_metrics.csv', index=False)
        print("\nNetwork metrics saved to network_metrics.csv")

if __name__ == '__main__':
    run_pipeline()