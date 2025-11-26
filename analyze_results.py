"""파이프라인 실행으로 생성된 모든 분기별 테스트 결과를 취합하고,
최종 통합 리포트(portfolio_report.csv)를 생성하는 스크립트
"""
import pandas as pd
import numpy as np
import os
import config

def analyze_results():
    """모든 테스트 결과 폴더를 순회하며 성과를 취합하고 네트워크 지표와 결합"""
    
    all_quarters = pd.period_range(start=config.START_QUARTER, end=config.END_QUARTER, freq='Q')
    num_test_sets = len(all_quarters) - 1
    
    period_results = []
    
    # 1. 네트워크 통계 데이터 로드 및 전처리
    try:
        net_stats = pd.read_csv('network_metrics.csv')
        # 지표 계산 (이미 계산되어 있을 수 있지만 안전하게 다시 계산하거나, 파일에 있는 값 그대로 사용)

        # run.py only saves: Quarter, Num_Nodes, Num_Edges, Num_Communities, Modularity
        # So we MUST calculate Density and Avg_Degree here.
        
        # 지표 계산
        net_stats['Network_Density'] = (2 * net_stats['Num_Edges']) / (net_stats['Num_Nodes'] * (net_stats['Num_Nodes'] - 1))
        net_stats['Avg_Degree'] = (2 * net_stats['Num_Edges']) / net_stats['Num_Nodes']
        
        # Quarter를 Period로 변환하여 매칭 준비 (Network Quarter -> Test Quarter)
        net_stats['Quarter_Period'] = pd.PeriodIndex(net_stats['Quarter'], freq='Q')
        net_stats['Test_Quarter'] = (net_stats['Quarter_Period'] + 1).astype(str)
        
        # 매칭을 위한 딕셔너리 생성 (Test_Quarter를 키로 사용)
        if 'Modularity' not in net_stats.columns:
             net_stats['Modularity'] = np.nan
             
        net_stats_map = net_stats.set_index('Test_Quarter')[['Network_Density', 'Avg_Degree', 'Modularity']].to_dict('index')
        
    except FileNotFoundError:
        print("Warning: network_statistics.csv not found. Please run run.py first.")
        return

    # 2. 모든 분기별 백테스트 결과 취합
    for i in range(num_test_sets):
        network_quarter = all_quarters[i]
        test_quarter = all_quarters[i+1]
        test_quarter_str = str(test_quarter)
        
        folder_name = f"Test_{i+1:02d}_({network_quarter}-{test_quarter})"
        file_path = os.path.join(config.TESTS_OUTPUT_DIR, folder_name, 'backtest_results.csv')
        
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File not found for {folder_name}. Skipping.")
            continue

        # 전략 포트폴리오와 무작위 포트폴리오 결과 분리
        main_results = df[df['portfolio_type'] != 'random'].set_index('portfolio_type')
        random_results = df[df['portfolio_type'] == 'random']

        if random_results.empty:
            continue
            
        # 해당 분기의 네트워크 지표 가져오기
        current_net_stats = net_stats_map.get(test_quarter_str, {'Network_Density': np.nan, 'Avg_Degree': np.nan, 'Modularity': np.nan})

        # 각 전략의 성과 분석
        for strategy_name in main_results.index:
            strategy_perf = main_results.loc[strategy_name]
            
            # Outperformance Rate (Top %)
            # Return: 높을수록 좋음 -> 랜덤보다 내가 높을 확률 (1 - percentile) 혹은 (랜덤 < 나) 비율
            # 여기서는 "랜덤 중 나보다 잘한 애들의 비율" (Top %)로 정의하는 게 직관적임 (1% = 최상위)
            # 즉, (Random > Strategy)의 비율
            outperformance_rate_return = (random_results['Cumulative_Return'] > strategy_perf['Cumulative_Return']).mean()
            
            # Volatility: 낮을수록 좋음 -> (Random < Strategy)의 비율 (내가 랜덤보다 변동성이 클 확률 = 나보다 변동성 작은 애들의 비율)
            # 즉, "랜덤 중 나보다 변동성이 작은(좋은) 애들의 비율"
            outperformance_rate_volatility = (random_results['Volatility'] < strategy_perf['Volatility']).mean()
            
            period_results.append({
                'Quarter': test_quarter_str,
                'Portfolio': strategy_name,
                'Return': strategy_perf['Cumulative_Return'],
                'Volatility': strategy_perf['Volatility'],
                'Top_Return': outperformance_rate_return,
                'Top_Volatility': outperformance_rate_volatility, # [추가] 변동성 상위 %
                'Network_Density': current_net_stats['Network_Density'],
                'Avg_Degree': current_net_stats['Avg_Degree'],
                'Modularity': current_net_stats['Modularity']
            })

    if not period_results:
        print("No results to analyze. Exiting.")
        return

    final_df = pd.DataFrame(period_results)

    # ---------------------------------------------------------
    # [수정] 리포트 분리 저장: 네트워크 지표 vs 포트폴리오 성과
    # ---------------------------------------------------------

    # 1. 네트워크 통계 저장 (모든 지표 포함)
    # net_stats는 이미 Density, Avg_Degree가 계산되어 있음
    # 저장할 컬럼 순서 정리
    net_cols = ['Quarter', 'Num_Nodes', 'Num_Edges', 'Num_Communities', 'Modularity', 'Network_Density', 'Avg_Degree']
    # net_stats에 있는 컬럼만 선택 (에러 방지)
    net_cols = [c for c in net_cols if c in net_stats.columns]
    
    net_output_path = 'network_metrics.csv'
    net_stats[net_cols].to_csv(net_output_path, index=False, float_format="%.4f")
    print(f"\nNetwork metrics saved to: {net_output_path}")

    # 2. 포트폴리오별 성과 리포트 (네트워크 지표 제외)
    perf_cols = ['Quarter', 'Return', 'Volatility', 'Top_Return', 'Top_Volatility']
    
    # Top 컬럼 포맷팅 (퍼센트 문자열로 변환)
    for col in ['Top_Return', 'Top_Volatility']:
        if col in final_df.columns:
             final_df[col] = final_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else x)

    # Max Eigenvector Portfolio Report
    max_df = final_df[final_df['Portfolio'] == 'max_eigenvector'].copy()
    max_df = max_df[perf_cols]
    max_output_path = 'max_centrality_portfolio.csv'
    max_df.to_csv(max_output_path, index=False, float_format="%.4f")
    print(f"Max Centrality portfolio report saved to: {max_output_path}")
    
    # Min Eigenvector Portfolio Report
    min_df = final_df[final_df['Portfolio'] == 'min_eigenvector'].copy()
    min_df = min_df[perf_cols]
    min_output_path = 'min_centrality_portfolio.csv'
    min_df.to_csv(min_output_path, index=False, float_format="%.4f")
    print(f"Min portfolio report saved to: {min_output_path}")
    
    # 미리보기 출력
    print("\n--- Max Centrality Report (Preview) ---")
    print(max_df.head())

    # [추가] 중간 파일(network_statistics.csv) 삭제
    if os.path.exists('network_statistics.csv'):
        os.remove('network_statistics.csv')
        # print("Intermediate file 'network_statistics.csv' deleted.") # 굳이 출력 안 함

if __name__ == '__main__':
    analyze_results()