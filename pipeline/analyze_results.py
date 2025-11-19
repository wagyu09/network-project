import pandas as pd
import numpy as np
import os

def analyze_results():
    all_quarters = pd.period_range(start='2017Q3', end='2025Q3', freq='Q')
    num_test_sets = len(all_quarters) - 3
    
    # Store metrics for each portfolio type
    metrics = {
        'min_centrality': {'returns': [], 'volatility': [], 'sharpe': [], 'mdd': []},
        'max_centrality': {'returns': [], 'volatility': [], 'sharpe': [], 'mdd': []},
        'random': {'returns': [], 'volatility': [], 'sharpe': [], 'mdd': []}
    }
    
    # Store win counts for each metric
    wins = {
        'Volatility': {'min_centrality': 0, 'max_centrality': 0, 'random': 0},
        'Sharpe_Ratio': {'min_centrality': 0, 'max_centrality': 0, 'random': 0},
        'Max_Drawdown': {'min_centrality': 0, 'max_centrality': 0, 'random': 0}
    }

    # Loop through all test folders to gather results
    for i in range(num_test_sets):
        full_period = all_quarters[i:i+4]
        folder_name = f"Test_{i+1:02d}_({full_period[0]}-{full_period[-1]})"
        file_path = os.path.join('pipeline', 'tests', folder_name, 'backtest_results.csv')
        
        try:
            df = pd.read_csv(file_path, index_col=0)
            
            # Map old index names to new keys for easier processing
            df.index = df.index.str.replace('_portfolio', '')
            
            # Gather metrics
            for p_key in metrics.keys():
                if p_key in df.index:
                    metrics[p_key]['returns'].append(df.loc[p_key, 'Cumulative_Return'])
                    metrics[p_key]['volatility'].append(df.loc[p_key, 'Volatility'])
                    metrics[p_key]['sharpe'].append(df.loc[p_key, 'Sharpe_Ratio'])
                    metrics[p_key]['mdd'].append(df.loc[p_key, 'Max_Drawdown'])
            
            # Compare metrics for this period if data is valid for all portfolios
            if df['Volatility'].notna().all():
                lowest_vol_winner = df['Volatility'].idxmin()
                wins['Volatility'][lowest_vol_winner] += 1
            
            if df['Sharpe_Ratio'].notna().all():
                highest_sharpe_winner = df['Sharpe_Ratio'].idxmax()
                wins['Sharpe_Ratio'][highest_sharpe_winner] += 1

            if df['Max_Drawdown'].notna().all():
                best_mdd_winner = df['Max_Drawdown'].idxmax() # Higher MDD is better (less negative)
                wins['Max_Drawdown'][best_mdd_winner] += 1

        except FileNotFoundError:
            print(f"File not found for {folder_name}. Skipping.")

    # --- Create Summary DataFrame ---
    summary_data = []
    portfolio_names = {
        'min_centrality': '최소 중심성 포트폴리오',
        'max_centrality': '최대 중심성 포트폴리오',
        'random': '랜덤 포트폴리오'
    }

    for p_key, p_name in portfolio_names.items():
        summary_data.append({
            'Portfolio': p_name,
            'Avg. Return': np.nanmean(metrics[p_key]['returns']),
            'Avg. Volatility': np.nanmean(metrics[p_key]['volatility']),
            'Avg. Sharpe Ratio': np.nanmean(metrics[p_key]['sharpe']),
            'Avg. Max Drawdown': np.nanmean(metrics[p_key]['mdd']),
            'Volatility Wins': wins['Volatility'][p_key],
            'Sharpe Ratio Wins': wins['Sharpe_Ratio'][p_key],
            'MDD Wins': wins['Max_Drawdown'][p_key]
        })
        
    summary_df = pd.DataFrame(summary_data).set_index('Portfolio')

    # --- Print and Save Results ---
    print("\n--- 전체 기간 통합 분석 결과 ---")
    print(summary_df.to_string(float_format="%.4f"))

    output_path = os.path.join('pipeline', 'final_summary_report.csv')
    summary_df.to_csv(output_path, float_format="%.4f")
    print(f"\n최종 분석 결과가 다음 파일에 저장되었습니다: {output_path}")

analyze_results()
