import pandas as pd
import numpy as np
from data_loader import DataLoader
from corr_calculator import _calculate_residuals 
from threshold import threshold
import community_detection
import network_visualizer

STARTDATE = '2024-01-01'
ENDDATE = '2025-01-01'

# 1. 데이터 로딩 및 전처리
data_loader = DataLoader(STARTDATE, ENDDATE)
daily, _, mkt_idx = data_loader.load_data() 

# 2. 잔차 상관관계 계산
corr_stats = _calculate_residuals(daily, mkt_idx)

# 3. 통계적 유의성에 기반한 엣지 필터링 (alpha=0.01, 상관계수 0.4 이상)
p_edges = threshold(corr_stats, alpha=0.01, c_min=0.4)

# 2024년 1분기 데이터만 선택
q1_edges = p_edges.loc['2024Q1'].reset_index()

# 상관계수 절댓값 평균 및 최솟값 출력
mean_abs_corr = q1_edges['Correlation'].abs().mean()
min_abs_corr = q1_edges['Correlation'].abs().min()
print(f"2024년 1분기 필터링된 엣지들의 상관계수 절댓값 평균: {mean_abs_corr:.4f}")
print(f"2024년 1분기 필터링된 엣지들의 상관계수 절댓값 최솟값: {min_abs_corr:.4f}")

# ... (previous code remains the same until partition detection) ...

# 4. 군집화: 양의 상관관계만 사용하여 커뮤니티 탐지
positive_edges = q1_edges[q1_edges['Correlation'] > 0]
G_community = community_detection.create_network_from_edges(positive_edges, weight_col='Correlation')
partition = community_detection.detect_communities(G_community, weight_col='Correlation', random_state=42)

# 5. 군집 간 평균 상관관계 계산
# 모든 고유한 커뮤니티 ID를 가져옵니다.
all_community_ids = sorted(list(set(partition.values())))

# 군집 간 평균 상관관계를 저장할 빈 DataFrame을 생성합니다.
inter_community_matrix = pd.DataFrame(0.0, index=all_community_ids, columns=all_community_ids)

# 각 엣지에 대해 커뮤니티 정보 추가
q1_edges['comm1'] = q1_edges['ticker1'].map(partition)
q1_edges['comm2'] = q1_edges['ticker2'].map(partition)

# 커뮤니티 쌍별로 평균 상관관계 계산
inter_edges = q1_edges.dropna(subset=['comm1', 'comm2']) # 파티션에 없는 노드 제외
inter_edges = inter_edges[inter_edges['comm1'] != inter_edges['comm2']]

# (comm1, comm2) 순서 정렬하여 중복 방지
inter_edges['comm_pair'] = inter_edges.apply(lambda row: tuple(sorted((int(row['comm1']), int(row['comm2'])))), axis=1)
avg_inter_corr_series = inter_edges.groupby('comm_pair')['Correlation'].mean()

# 계산된 평균 상관관계를 매트릭스에 채워넣습니다.
for (c1, c2), corr_val in avg_inter_corr_series.items():
    inter_community_matrix.loc[c1, c2] = corr_val
    inter_community_matrix.loc[c2, c1] = corr_val # 대칭적으로 채움

print("\n--- 군집 간 평균 상관관계 매트릭스 ---")
print(inter_community_matrix)

# 군집 간 평균 상관관계 매트릭스를 CSV 파일로 저장
output_csv_filename = "inter_community_correlation.csv"
inter_community_matrix.to_csv(output_csv_filename)
print(f"\n군집 간 평균 상관관계 매트릭스가 '{output_csv_filename}' 파일로 저장되었습니다.")

# 6. 시각화: 모든 유의미한 엣지(양수/음수)를 사용하여 네트워크 생성
#    (노드 목록이 군집화와 시각화 그래프 간에 동일하도록 보장)
all_nodes = list(G_community.nodes())
G_visual = community_detection.create_network_from_edges(q1_edges)
G_visual.add_nodes_from(all_nodes) # 군집화에 사용된 모든 노드가 시각화에 포함되도록 보장

# 7. 커뮤니티 및 엣지 종류(양/음)를 포함한 네트워크 시각화
network_visualizer.visualize_network(G_visual, partition)
