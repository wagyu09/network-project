"""파이프라인 실행으로 생성된 모든 분기별 테스트 결과를 취합하고,
최종 요약 리포트를 생성하는 스크립트
"""
import pandas as pd
import numpy as np
import os
import sys
import config

def analyze_results():
    """모든 테스트 결과 폴더를 순회하며 `backtest_results.csv` 파일을 분석

    각 분기별로 전략 포트폴리오의 성과를 무작위 포트폴리오의 성과 분포와 비교하여
    무작위 포트폴리오가 더 나은 성과를 보인 비율(Outperformance Rate)을 계산하고,
    전체 기간에 대한 평균 성과를 구해 최종 리포트를 생성
    """
    all_quarters = pd.period_range(start=config.START_QUARTER, end=config.END_QUARTER, freq='Q')
    num_test_sets = len(all_quarters) - 1
    
    period_results = []

    for i in range(num_test_sets):
        network_quarter = all_quarters[i]
        test_quarter = all_quarters[i+1]
        folder_name = f"Test_{i+1:02d}_({network_quarter}-{test_quarter})"
        file_path = os.path.join(config.TESTS_OUTPUT_DIR, folder_name, 'backtest_results.csv')
        
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File not found for {folder_name}. Skipping.")
            continue

        main_results = df[df['portfolio_type'] != 'random'].set_index('portfolio_type')
        random_results = df[df['portfolio_type'] == 'random']

        if random_results.empty:
            print(f"No random portfolio data for {folder_name}. Skipping analysis for this period.")
            continue

        random_means = random_results.mean(numeric_only=True)

        for strategy_name in main_results.index:
            strategy_perf = main_results.loc[strategy_name]
            
            # 무작위 포트폴리오가 전략 포트폴리오보다 더 나은 성과를 보인 비율 계산
            outperformance_rate_return = (random_results['Cumulative_Return'] > strategy_perf['Cumulative_Return']).mean()
            outperformance_rate_sharpe = (random_results['Sharpe_Ratio'] > strategy_perf['Sharpe_Ratio']).mean()
            outperformance_rate_mdd = (random_results['Max_Drawdown'] > strategy_perf['Max_Drawdown']).mean()
            outperformance_rate_volatility = (random_results['Volatility'] < strategy_perf['Volatility']).mean()

            period_results.append({
                'Portfolio': strategy_name.replace('_portfolio', ''),
                'Return': strategy_perf['Cumulative_Return'],
                'Volatility': strategy_perf['Volatility'],
                'Sharpe Ratio': strategy_perf['Sharpe_Ratio'],
                'Max Drawdown': strategy_perf['Max_Drawdown'],
                'Random Mean Return': random_means['Cumulative_Return'],
                'Random Mean Volatility': random_means['Volatility'],
                'Random Mean Sharpe': random_means['Sharpe_Ratio'],
                'Random Mean MDD': random_means['Max_Drawdown'],
                'Random_Outperformance_Rate (Return)': outperformance_rate_return,
                'Random_Outperformance_Rate (Volatility)': outperformance_rate_volatility,
                'Random_Outperformance_Rate (Sharpe)': outperformance_rate_sharpe,
                'Random_Outperformance_Rate (MDD)': outperformance_rate_mdd,
            })

    if not period_results:
        print("No results to analyze. Exiting.")
        return

    # --- 최종 요약 데이터프레임 생성 ---
    final_df = pd.DataFrame(period_results)
    
    # 포트폴리오별로 모든 기간의 지표를 평균
    summary_df = final_df.groupby('Portfolio').mean()
    
    # 컬럼 이름의 접두사를 'Avg_'로 변경하여 '평균' 값임을 명시
    summary_df = summary_df.rename(columns=lambda c: 'Avg_' + c if 'Random_Outperformance_Rate' in c else c)

    # --- 결과 출력 및 저장 ---
    print("\n--- 전체 기간 통합 분석 결과 ---")
    print(summary_df.to_string(float_format="%.4f"))

    output_path = config.FINAL_REPORT_PATH
    summary_df.to_csv(output_path, float_format="%.4f")
    print(f"\n최종 분석 결과가 다음 파일에 저장되었습니다: {output_path}")


if __name__ == '__main__':
    analyze_results()