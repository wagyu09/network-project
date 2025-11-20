import pandas as pd
import numpy as np
import os

def analyze_results():
    all_quarters = pd.period_range(start='2020Q1', end='2025Q3', freq='Q')
    num_test_sets = len(all_quarters) - 1
    
    # Store detailed results for each period
    period_results = []

    # Loop through all test folders to gather and analyze results
    for i in range(num_test_sets):
        network_quarter = all_quarters[i]
        test_quarter = all_quarters[i+1]
        folder_name = f"Test_{i+1:02d}_({network_quarter}-{test_quarter})"
        file_path = os.path.join('pipeline', 'tests', folder_name, 'backtest_results.csv')
        
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File not found for {folder_name}. Skipping.")
            continue

        # Separate main strategies from the random distribution
        main_results = df[df['portfolio_type'] != 'random'].set_index('portfolio_type')
        random_results = df[df['portfolio_type'] == 'random']

        if random_results.empty:
            print(f"No random portfolio data for {folder_name}. Skipping analysis for this period.")
            continue

        # Calculate mean performance of the random distribution
        random_means = random_results.mean(numeric_only=True)

        # Analyze each main strategy against the random distribution
        for strategy_name in main_results.index:
            strategy_perf = main_results.loc[strategy_name]
            
            # Calculate p-values (fraction of random portfolios that performed better)
            # For Return, Sharpe, and MDD, higher is better
            p_value_return = (random_results['Cumulative_Return'] > strategy_perf['Cumulative_Return']).mean()
            p_value_sharpe = (random_results['Sharpe_Ratio'] > strategy_perf['Sharpe_Ratio']).mean()
            p_value_mdd = (random_results['Max_Drawdown'] > strategy_perf['Max_Drawdown']).mean()
            
            # For Volatility, lower is better
            p_value_volatility = (random_results['Volatility'] < strategy_perf['Volatility']).mean()

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
                'p-value (Return)': p_value_return,
                'p-value (Volatility)': p_value_volatility,
                'p-value (Sharpe)': p_value_sharpe,
                'p-value (MDD)': p_value_mdd,
            })

    if not period_results:
        print("No results to analyze. Exiting.")
        return

    # --- Create Final Summary DataFrame ---
    final_df = pd.DataFrame(period_results)
    
    # Calculate the average of all metrics across all periods, grouped by portfolio
    summary_df = final_df.groupby('Portfolio').mean()

    # --- Print and Save Results ---
    print("\n--- 전체 기간 통합 분석 결과 ---")
    print(summary_df.to_string(float_format="%.4f"))

    output_path = os.path.join(os.getcwd(), 'final_summary_report.csv')
    summary_df.to_csv(output_path, float_format="%.4f")
    print(f"\n최종 분석 결과가 다음 파일에 저장되었습니다: {output_path}")


if __name__ == '__main__':
    analyze_results()
