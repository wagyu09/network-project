import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_network_metrics(input_filename='network_metrics.csv', output_filename='network_metrics_timeseries.png'):
    """
    network_metrics.csv 파일을 읽어 주요 네트워크 지표들의 시계열 변화를
    하나의 그래프로 시각화하고 이미지 파일로 저장합니다.

    - Modularity, Network Density는 왼쪽 Y축을 사용합니다.
    - Average Degree는 오른쪽 Y축을 사용하여 스케일 차이를 보정합니다.
    """
    try:
        # CSV 파일 로드
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"오류: '{input_filename}' 파일을 찾을 수 없습니다.")
        print("먼저 run.py를 실행하여 network_metrics.csv 파일을 생성해야 합니다.")
        return

    # --- 시각화 ---
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # X축 설정 (Quarter)
    quarters = df['Quarter']
    
    # 왼쪽 Y축 (ax1) - Modularity, Density
    color1 = 'tab:blue'
    ax1.set_xlabel('Quarter', fontsize=14)
    ax1.set_ylabel('Modularity / Density', color=color1, fontsize=14)
    line1, = ax1.plot(quarters, df['Modularity'], color=color1, marker='o', linestyle='-', label='Modularity')
    line2, = ax1.plot(quarters, df['network_density'], color='tab:cyan', marker='x', linestyle='--', label='Network Density')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.tick_params(axis='x', rotation=45) # x축 라벨 회전

    # 오른쪽 Y축 (ax2) - Average Degree
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Average Degree', color=color2, fontsize=14)
    line3, = ax2.plot(quarters, df['avg_degree'], color=color2, marker='s', linestyle='-', label='Average Degree')
    ax2.tick_params(axis='y', labelcolor=color2)

    # 그래프 제목 및 범례
    plt.title('Network Metrics Over Time', fontsize=18)
    fig.tight_layout() # 레이아웃 최적화

    # 범례 합치기
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # 그리드 추가
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 이미지 파일로 저장
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"네트워크 지표 시계열 그래프가 '{output_filename}' 파일로 저장되었습니다.")

if __name__ == '__main__':
    plot_network_metrics()
