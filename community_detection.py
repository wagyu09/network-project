import pandas as pd
import networkx as nx
import community as community_louvain

def create_network_from_edges(edges_df: pd.DataFrame, weight_col: str = 'Correlation', edge_attrs=None):
    """
    엣지 목록 DataFrame으로부터 NetworkX 그래프를 생성합니다.

    :param edges_df: 'ticker1', 'ticker2' 및 가중치 컬럼을 포함하는 DataFrame.
    :param weight_col: 엣지 가중치로 사용할 컬럼 이름.
    :param edge_attrs: 엣지 속성으로 추가할 컬럼 리스트.
    :return: NetworkX 그래프 객체.
    """
    G = nx.from_pandas_edgelist(
        edges_df,
        source='ticker1',
        target='ticker2',
        edge_attr=edge_attrs if edge_attrs else [weight_col]
    )
    return G

def detect_communities(G: nx.Graph, weight_col: str = 'Correlation', random_state: int = None):
    """
    주어진 그래프에 대해 Louvain 알고리즘을 사용하여 커뮤니티를 탐지합니다.

    :param G: NetworkX 그래프 객체.
    :param weight_col: 커뮤니티 탐지에 사용할 엣지 가중치 컬럼.
    :param random_state: 재현성을 위한 시드 값.
    :return: 노드를 커뮤니티 ID에 매핑하는 딕셔너리 (파티션).
    """
    partition = community_louvain.best_partition(G, weight=weight_col, random_state=random_state)
    return partition

def calculate_inter_community_correlation(all_edges: pd.DataFrame, partition: dict):
    """
    군집 정보를 사용하여 군집 간 평균 상관관계를 계산합니다.

    :param all_edges: 'ticker1', 'ticker2', 'Correlation' 컬럼을 포함하는 모든 엣지 DataFrame.
    :param partition: 노드를 커뮤니티 ID에 매핑하는 딕셔너리.
    :return: 군집 간 평균 상관관계를 담은 DataFrame (매트릭스).
    """
    if not partition:
        return pd.DataFrame()

    # 모든 고유한 커뮤니티 ID를 가져옵니다.
    all_community_ids = sorted(list(set(partition.values())))
    
    # 군집 간 평균 상관관계를 저장할 빈 DataFrame을 생성합니다.
    inter_community_matrix = pd.DataFrame(0.0, index=all_community_ids, columns=all_community_ids)

    # 각 엣지에 대해 커뮤니티 정보 추가
    edges_with_comm = all_edges.copy()
    edges_with_comm['comm1'] = edges_with_comm['ticker1'].map(partition)
    edges_with_comm['comm2'] = edges_with_comm['ticker2'].map(partition)

    # 파티션에 없는 노드 관련 엣지 제외 및 군집 내부 엣지 제외
    inter_edges = edges_with_comm.dropna(subset=['comm1', 'comm2'])
    inter_edges = inter_edges[inter_edges['comm1'] != inter_edges['comm2']]
    
    if inter_edges.empty:
        return inter_community_matrix

    # (comm1, comm2) 순서 정렬하여 중복 방지 및 평균 계산
    inter_edges['comm_pair'] = inter_edges.apply(lambda row: tuple(sorted((int(row['comm1']), int(row['comm2'])))), axis=1)
    avg_inter_corr_series = inter_edges.groupby('comm_pair')['Correlation'].mean()

    # 계산된 평균 상관관계를 매트릭스에 채워넣습니다.
    for (c1, c2), corr_val in avg_inter_corr_series.items():
        inter_community_matrix.loc[c1, c2] = corr_val
        inter_community_matrix.loc[c2, c1] = corr_val # 대칭적으로 채움
        
    return inter_community_matrix