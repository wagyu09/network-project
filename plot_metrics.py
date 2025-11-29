"""네트워크 지표 시계열 변화를 시각화하는 스크립트"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import config

def plot_network_metrics():
    metrics_path = os.path.join(config.SUMMARY_DIR, 'network_metrics.csv')
    if not os.path.exists(metrics_path):
        print("Metrics file not found.")
        return
        
    df = pd.read_csv(metrics_path)
    df['Quarter'] = df['Quarter'].astype(str)
    
    # 그래프 설정
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 왼쪽 축: Modularity, Density
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('Modularity & Density')
    
    line1 = ax1.plot(df['Quarter'], df['Modularity'], marker='o', label='Modularity', color='tab:blue')
    line2 = ax1.plot(df['Quarter'], df['network_density'], marker='s', label='Density', color='tab:orange')
    
    ax1.tick_params(axis='y')
    plt.xticks(rotation=45)

    # 오른쪽 축: Avg Degree (스케일이 다르므로)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Degree')
    line3 = ax2.plot(df['Quarter'], df['avg_degree'], marker='^', label='Avg Degree', color='tab:green')
    ax2.tick_params(axis='y')

    # 범례 합치기
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title('Network Metrics Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(config.FIGURES_DIR, 'network_metrics_timeseries.png')
    plt.savefig(save_path)
    print(f"네트워크 지표 시계열 그래프가 '{save_path}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    plot_network_metrics()
