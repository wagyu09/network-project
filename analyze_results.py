"""포트폴리오 백테스팅 결과를 종합하여 분석하는 스크립트"""
import os
import pandas as pd
import glob
import config

def analyze_backtest_results():
    results_dir = config.QUARTERLY_DIR
    summary_dir = config.SUMMARY_DIR
    
    # 모든 분기의 백테스트 결과 수집
    pattern = os.path.join(results_dir, "*", "portfolios", "backtest_results.csv")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("No backtest results found.")
        return

    # 데이터 취합
    all_res = []
    for f in files:
        q = f.split(os.path.sep)[-3]
        df = pd.read_csv(f)
        df['Quarter'] = q
        all_res.append(df)
        
    full_df = pd.concat(all_res, ignore_index=True)
    
    # 전략 매핑
    strategy_map = {
        'max_eigenvector': 'max_centrality_portfolio',
        'min_eigenvector': 'min_centrality_portfolio',
        'weighted_sector': 'weighted_portfolio'
    }
    
    for strat_key, filename in strategy_map.items():
        strat_df = full_df[full_df['portfolio_type'] == strat_key].copy()
        
        # 각 분기별 Random 포트폴리오와의 비교 (Top %)
        top_metrics = []
        for q in strat_df['Quarter'].unique():
            rand_df = full_df[(full_df['Quarter'] == q) & (full_df['portfolio_type'] == 'random')]
            target = strat_df[strat_df['Quarter'] == q].iloc[0]
            
            n_total = len(rand_df)
            if n_total == 0: continue
            
            rank_ret = (rand_df['Cumulative_Return'] > target['Cumulative_Return']).sum()
            rank_vol = (rand_df['Volatility'] < target['Volatility']).sum()
            rank_shp = (rand_df['Sharpe_Ratio'] > target['Sharpe_Ratio']).sum()
            
            top_metrics.append({
                'Quarter': q,
                'Top_Return': rank_ret / n_total * 100,
                'Top_Volatility': rank_vol / n_total * 100,
                'Top_Sharpe': rank_shp / n_total * 100
            })
            
        top_df = pd.DataFrame(top_metrics)
        final_strat = pd.merge(strat_df, top_df, on='Quarter')
        
        # 컬럼명 변경
        final_strat.rename(columns={'Cumulative_Return': 'Return'}, inplace=True)
        
        # 필요한 컬럼만 선택
        cols = ['Quarter', 'Return', 'Volatility', 'Sharpe_Ratio', 'Top_Return', 'Top_Volatility', 'Top_Sharpe']
        final_strat = final_strat[cols]
        
        # [복구] 평균 행 추가 로직
        numeric_cols = ['Return', 'Volatility', 'Sharpe_Ratio', 'Top_Return', 'Top_Volatility', 'Top_Sharpe']
        avg_row = final_strat[numeric_cols].mean().to_dict()
        avg_row['Quarter'] = 'Average'
        
        # DataFrame에 평균 행 추가
        final_strat = pd.concat([final_strat, pd.DataFrame([avg_row])], ignore_index=True)
        
        # Top 지표들 % 포맷팅 (평균 계산 후 포맷팅해야 함)
        for col in ['Top_Return', 'Top_Volatility', 'Top_Sharpe']:
            final_strat[col] = final_strat[col].apply(lambda x: "{:.1f}%".format(x))
        
        # 결과 저장
        save_path = os.path.join(summary_dir, f'{filename}.csv')
        final_strat.to_csv(save_path, index=False)
        
        print(f"Saved {filename}.csv")
        print(f"\n--- {filename} Report (Tail) ---")
        print(final_strat.tail().to_string(index=False))

if __name__ == "__main__":
    analyze_backtest_results()
