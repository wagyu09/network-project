"네트워크 분석, 커뮤니티 탐지 및 시각화와 관련된 함수들을 관리하는 모듈"
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def calculate_centrality(G: nx.Graph) -> dict:
    """그래프 내 각 노드의 중심성(Degree Centrality)을 계산

    Args:
        G (nx.Graph): NetworkX 그래프 객체

    Returns:
        dict: 각 노드의 이름을 키로, 중심성 값을 값으로 하는 딕셔너리
            그래프에 노드가 없을 경우 빈 딕셔너리 반환
    """
    if not G.nodes():
        return {}
    return nx.degree_centrality(G)

def create_network_from_edges(edges_df: pd.DataFrame, weight_col: str = 'Correlation', edge_attrs=None) -> nx.Graph:
    """엣지 목록 DataFrame으로부터 NetworkX 그래프를 생성

    Args:
        edges_df (pd.DataFrame): 'ticker1'(소스), 'ticker2'(타겟) 및
            가중치 컬럼을 포함하는 데이터프레임
        weight_col (str): 엣지 가중치로 사용할 컬럼 이름. 기본값은 'Correlation'
        edge_attrs (list, optional): 엣지 속성으로 추가할 컬럼 리스트
            지정하지 않으면 weight_col만 추가

    Returns:
        nx.Graph: 생성된 NetworkX 그래프 객체
    """
    G = nx.from_pandas_edgelist(
        edges_df,
        source='ticker1',
        target='ticker2',
        edge_attr=edge_attrs if edge_attrs else [weight_col]
    )
    return G

def detect_communities(G: nx.Graph, weight_col: str = 'Correlation', random_state: int = None) -> dict:
    """주어진 그래프에 대해 Louvain 알고리즘을 사용하여 커뮤니티를 탐지

    Args:
        G (nx.Graph): NetworkX 그래프 객체
        weight_col (str): 커뮤니티 탐지에 사용할 엣지 가중치 컬럼. 기본값은 'Correlation'
        random_state (int, optional): 재현성을 위한 시드 값. 지정하지 않으면 무작위

    Returns:
        dict: 노드를 커뮤니티 ID에 매핑하는 딕셔너리 (파티션)
    """
    partition = community_louvain.best_partition(G, weight=weight_col, random_state=random_state)
    return partition

def calculate_inter_community_correlation(all_edges: pd.DataFrame, partition: dict) -> pd.DataFrame:
    """군집 정보를 사용하여 군집 간 평균 상관관계를 계산

    Args:
        all_edges (pd.DataFrame): 'ticker1', 'ticker2', 'Correlation' 컬럼을 포함하는
            모든 엣지 데이터프레임
        partition (dict): 노드를 커뮤니티 ID에 매핑하는 딕셔너리

    Returns:
        pd.DataFrame: 군집 간 평균 상관관계를 담은 데이터프레임 (매트릭스 형태)
            파티션이 비어있으면 빈 데이터프레임 반환
    """
    if not partition:
        return pd.DataFrame()

    # 모든 고유한 커뮤니티 ID를 가져옴
    all_community_ids = sorted(list(set(partition.values())))
    
    # 군집 간 평균 상관관계를 저장할 빈 데이터프레임 생성
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

    # 계산된 평균 상관관계를 매트릭스에 채워 넣음
    for (c1, c2), corr_val in avg_inter_corr_series.items():
        inter_community_matrix.loc[c1, c2] = corr_val
        inter_community_matrix.loc[c2, c1] = corr_val # 대칭적으로 채움
        
    return inter_community_matrix

def visualize_network(G: nx.Graph, partition: dict, output_filename: str = 'network_visualization.png'):
    """커뮤니티가 탐지된 네트워크를 시각화하고 파일로 저장

    Args:
        G (nx.Graph): NetworkX 그래프 객체
        partition (dict): 노드와 커뮤니티 ID 매핑 딕셔너리
        output_filename (str): 저장할 이미지 파일 이름. 기본값은 'network_visualization.png'

    Returns:
        None: 시각화된 이미지를 파일로 저장
    """
    if not G.nodes():
        print("시각화할 노드가 없음")
        return

    # 커뮤니티별 색상 지정을 위한 준비 (색상 구분이 명확한 'tab20' 사용)
    # partition에 없는 노드(예: 음의 엣지만 가진 노드)는 회색으로 처리
    if partition:
        num_communities = len(set(partition.values()))
        # 20개 이상의 커뮤니티가 있을 경우를 대비해 여러 컬러맵을 합쳐 사용
        if num_communities > 20:
            cmap1 = cm.get_cmap('tab20', 20)
            cmap2 = cm.get_cmap('tab20b', 20)
            combined_colors = cmap1.colors + cmap2.colors
            cmap = lambda i: combined_colors[i % len(combined_colors)]
        else:
            cmap_tab20 = cm.get_cmap('tab20', num_communities)
            cmap = lambda i: cmap_tab20.colors[i]
    
    colors = []
    for node in G.nodes():
        if node in partition:
            colors.append(cmap(partition[node]))
        else:
            colors.append('grey') # 군집에 속하지 않는 노드 색상

    # 레이아웃을 위한 엣지 가중치 설정 (같은 군집은 강하게, 다른 군집은 약하게)
    for u, v, data in G.edges(data=True):
        # partition에 없는 노드들 간의 엣지 처리
        u_comm = partition.get(u)
        v_comm = partition.get(v)
        if u_comm is not None and u_comm == v_comm:
            G.edges[u,v]['layout_weight'] = 1.0 # 같은 군집
        else:
            G.edges[u,v]['layout_weight'] = 0.05 # 다른 군집 또는 군집 미소속 노드와의 연결

    # 네트워크 레이아웃 설정 (spring_layout으로 변경 및 가중치, 거리 조정)
    # k 값을 조절하여 노드 간의 기본 거리를 설정
    k_val = 1.5 / np.sqrt(len(G.nodes()))
    pos = nx.spring_layout(G, weight='layout_weight', k=k_val, iterations=100, seed=42)

    # 엣지 색상 및 두께 설정
    edge_colors = []
    edge_widths = []
    for u, v, data in G.edges(data=True):
        corr = data.get('Correlation', 0)
        edge_colors.append('lightblue' if corr > 0 else 'lightcoral')
        edge_widths.append(abs(corr) * 2) # 상관관계 절댓값에 비례하는 엣지 두께

    # 노드 크기를 디그리(연결 수)에 비례하도록 설정
    degrees = [val for (node, val) in G.degree()]
    node_sizes = [20 + d * 15 for d in degrees] # 기본 크기 + (디그리 * 스케일)

    plt.figure(figsize=(20, 20))

    # 노드, 엣지, 라벨 그리기
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, font_size=6, font_family='DejaVu Sans')

    num_communities = len(set(partition.values()))
    title = (
        f"Stock Network with Louvain Community Detection\n"
        f"(Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, Communities: {num_communities})"
    )
    plt.title(title, size=25)
    plt.axis('off')
    
    # 이미지 파일로 저장
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close() # GUI 창이 뜨지 않도록 닫아줌
    print(f"네트워크 시각화가 '{output_filename}' 파일로 저장되었습니다")
