"""파이프라인 실행으로 생성된 모든 분기별 테스트 결과를 취합하고,
최종 요약 리포트를 생성하는 스크립트
"""
import pandas as pd
import numpy as np
import os
import sys
import statsmodels.api as sm
from scipy.stats import ttest_1samp
import config

def analyze_results():
    """모든 테스트 결과 폴더를 순회하며 `backtest_results.csv` 파일을 분석

    각 분기별 전략 포트폴리오의 성과를 다수의 무작위 포트폴리오 성과 분포와
    비교하여, 전략의 우수성을 통계적으로 검증하고 최종 리포트를 생성함
    """
    all_quarters = pd.period_range(start=config.START_QUARTER, end=config.END_QUARTER, freq='Q')
    num_test_sets = len(all_quarters) - 1
    
    period_results = []
    all_random_results = []

    # 1. 모든 분기별 백테스트 결과 취합
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

        # 전략 포트폴리오와 무작위 포트폴리오 결과 분리
        main_results = df[df['portfolio_type'] != 'random'].set_index('portfolio_type')
        random_results = df[df['portfolio_type'] == 'random']

        if random_results.empty:
            print(f"No random portfolio data for {folder_name}. Skipping analysis for this period.")
            continue
            
        all_random_results.append(random_results)

        # 각 전략의 성과가 무작위 포트폴리오 분포에서 상위 몇 %에 위치하는지 계산
        for strategy_name in main_results.index:
            strategy_perf = main_results.loc[strategy_name]
            
            # Outperformance Rate: 무작위 포트폴리오가 전략보다 더 나은 성과를 보인 비율 (낮을수록 좋음)
            outperformance_rate_return = (random_results['Cumulative_Return'] > strategy_perf['Cumulative_Return']).mean()
            outperformance_rate_sharpe = (random_results['Sharpe_Ratio'] > strategy_perf['Sharpe_Ratio']).mean()
            outperformance_rate_mdd = (random_results['Max_Drawdown'] > strategy_perf['Max_Drawdown']).mean() # MDD는 음수이므로 값이 클수록(0에 가까울수록) 좋음
            outperformance_rate_volatility = (random_results['Volatility'] < strategy_perf['Volatility']).mean() # Volatility는 낮을수록 좋음

            period_results.append({
                'Portfolio': strategy_name.replace('_portfolio', ''),
                'Return': strategy_perf['Cumulative_Return'],
                'Volatility': strategy_perf['Volatility'],
                'Sharpe': strategy_perf['Sharpe_Ratio'],
                'MDD': strategy_perf['Max_Drawdown'],
                'Outperformance_Rate_Return': outperformance_rate_return,
                'Outperformance_Rate_Volatility': outperformance_rate_volatility,
                'Outperformance_Rate_Sharpe': outperformance_rate_sharpe,
                'Outperformance_Rate_MDD': outperformance_rate_mdd,
            })

    if not period_results:
        print("No results to analyze. Exiting.")
        return

    final_df = pd.DataFrame(period_results)
    
    # 2. 포트폴리오별 최종 요약 리포트 생성
    summary_list = []
    
    for portfolio_name, group in final_df.groupby('Portfolio'):
        
        # 2-1. 각 지표별 p-value 계산 (Newey-West HAC 추정)
        # 롤링 윈도우로 인한 시계열 자기상관을 보정하기 위해 사용
        p_values = {}
        for metric in ['Return', 'Volatility', 'Sharpe', 'MDD']:
            rates = group[f'Outperformance_Rate_{metric}'].tolist()
            
            # Y = (관측값 - 귀무가설 평균), H0: E[rates] = 0.5
            Y = np.array(rates) - 0.5
            # X = 상수항 (Intercept)
            X = np.ones(len(Y))
            
            # OLS 모델 적합 (Newey-West 표준오차 적용, 시차=1)
            model = sm.OLS(Y, X)
            results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})
            
            t_stat = results.tvalues[0]
            p_value_two_sided = results.pvalues[0]
            
            # 단측 검정 (H1: E[rates] < 0.5) p-value 계산
            # t-통계량이 음수일 때만 효과가 있는 방향(더 나은 성과)이므로 p-value/2
            if t_stat < 0:
                p_value_one_sided = p_value_two_sided / 2
            else: # t-통계량이 양수면 H1과 반대 방향이므로 유의하지 않음
                p_value_one_sided = 1 - (p_value_two_sided / 2)
            
            p_values[metric] = p_value_one_sided

        # 2-2. 포트폴리오별 평균 성과 및 p-value 집계
        summary_list.append({
            'Portfolio': portfolio_name,
            'Return': group['Return'].mean(),
            'Volatility': group['Volatility'].mean(),
            'Sharpe': group['Sharpe'].mean(),
            'MDD': group['MDD'].mean(),
            'Top % (Return)': group['Outperformance_Rate_Return'].mean(),
            'p-value (Return)': p_values['Return'],
            'Top % (Volatility)': group['Outperformance_Rate_Volatility'].mean(),
            'p-value (Volatility)': p_values['Volatility'],
            'Top % (Sharpe)': group['Outperformance_Rate_Sharpe'].mean(),
            'p-value (Sharpe)': p_values['Sharpe'],
            'Top % (MDD)': group['Outperformance_Rate_MDD'].mean(),
            'p-value (MDD)': p_values['MDD'],
        })
        
    summary_df = pd.DataFrame(summary_list).set_index('Portfolio')

    # 3. 무작위 포트폴리오의 전체 기간 평균 성과 계산 및 추가
    if all_random_results:
        combined_random_df = pd.concat(all_random_results)
        # 각 분기별 1000개 랜덤 포트폴리오의 평균을 먼저 구하고, 그 분기별 평균들의 전체 평균을 계산
        avg_random_perf = combined_random_df.groupby(np.arange(len(combined_random_df)) // config.NUM_RANDOM_PORTFOLIOS).mean(numeric_only=True).mean()
        
        random_avg_row = pd.DataFrame({
            'Return': [avg_random_perf['Cumulative_Return']],
            'Volatility': [avg_random_perf['Volatility']],
            'Sharpe': [avg_random_perf['Sharpe_Ratio']],
            'MDD': [avg_random_perf['Max_Drawdown']]
        }, index=['random_avg'])
        
        summary_df = pd.concat([summary_df, random_avg_row])
    
    # 4. 최종 데이터프레임 형식 정리
    # Top % 컬럼들을 백분율 형식으로 변환
    for col in summary_df.columns:
        if 'Top %' in col:
            summary_df[col] = summary_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else x)

    # 컬럼 순서 재배치
    column_order = [
        'Return', 'Volatility', 'Sharpe', 'MDD',
        'Top % (Return)', 'p-value (Return)',
        'Top % (Volatility)', 'p-value (Volatility)',
        'Top % (Sharpe)', 'p-value (Sharpe)',
        'Top % (MDD)', 'p-value (MDD)'
    ]
    summary_df = summary_df.reindex(columns=column_order)

    # --- 최종 결과 출력 및 저장 ---
    print("\n--- Final Summary Report ---")
    print(summary_df.to_string(float_format="%.4f"))

    output_path = config.FINAL_REPORT_PATH
    summary_df.to_csv(output_path, float_format="%.4f")
    print(f"\nFinal summary report saved to: {output_path}")

if __name__ == '__main__':
    analyze_results()