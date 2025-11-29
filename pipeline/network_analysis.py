"네트워크 분석, 커뮤니티 탐지 및 시각화와 관련된 함수들을 관리하는 모듈"
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def calculate_centrality(G: nx.Graph) -> dict:
    """그래프 내 각 노드의 중심성(Eigenvector Centrality)을 계산

    Args:
        G (nx.Graph): NetworkX 그래프 객체

    Returns:
        dict: 각 노드의 이름을 키로, 중심성 값을 값으로 하는 딕셔너리
            그래프에 노드가 없을 경우 빈 딕셔너리 반환
    """
    if not G.nodes():
        return {}
    
    try:
        # 가중치를 반영한 아이겐벡터 중심성 계산
        # max_iter를 충분히 늘려 수렴 가능성 높임
        return nx.eigenvector_centrality(G, weight='Correlation', max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("  Warning: Eigenvector centrality did not converge. Falling back to degree centrality.")
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

    # 커뮤니티별 색상 지정을 위한 준비 (색상 구분이 명확한 'tab20', 'tab20b', 'tab20c' 합쳐 사용)
    # 60개 이상의 커뮤니티를 지원하기 위해 여러 컬러맵을 합침
    cmap_tab20 = cm.get_cmap('tab20', 20)
    cmap_tab20b = cm.get_cmap('tab20b', 20)
    cmap_tab20c = cm.get_cmap('tab20c', 20)
    combined_colors_list = list(cmap_tab20.colors) + list(cmap_tab20b.colors) + list(cmap_tab20c.colors)
    
    color_map_func = lambda i: combined_colors_list[i % len(combined_colors_list)]
    
    colors = []
    for node in G.nodes():
        if node in partition:
            colors.append(color_map_func(partition[node]))
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

def visualize_degree_histogram(G: nx.Graph, output_filename: str):
    """네트워크의 디그리 분포 히스토그램을 시각화하고 파일로 저장

    Args:
        G (nx.Graph): NetworkX 그래프 객체
        output_filename (str): 저장할 이미지 파일 이름
    """
    if not G.nodes():
        print("디그리 분포를 계산할 노드가 없음")
        return

    degrees = [G.degree(n) for n in G.nodes()]
    
    plt.figure(figsize=(12, 8))
    plt.hist(degrees, bins='auto', color='skyblue', edgecolor='black', alpha=0.7)
    
    avg_degree = np.mean(degrees)
    plt.axvline(avg_degree, color='red', linestyle='dashed', linewidth=2)
    
    plt.title(f'Degree Distribution Histogram (Avg Degree: {avg_degree:.2f})', fontsize=18)
    plt.xlabel('Degree', fontsize=14)
    plt.ylabel('Number of Nodes', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # y축을 로그 스케일로 표시하여 멱함수 분포 확인
    plt.yscale('log')
    
    plt.minorticks_off()
    
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"디그리 분포 히스토그램이 '{output_filename}' 파일로 저장되었습니다")

def visualize_degree_ccdf(G: nx.Graph, output_filename: str):
    """네트워크의 디그리 분포의 상보적 누적 분포 함수(CCDF)를 시각화하고 파일로 저장

    Args:
        G (nx.Graph): NetworkX 그래프 객체
        output_filename (str): 저장할 이미지 파일 이름
    """
    if not G.nodes():
        print("CCDF를 계산할 노드가 없음")
        return

    degrees = np.array([G.degree(n) for n in G.nodes()])
    degree_counts = np.bincount(degrees)
    ccdf = 1 - np.cumsum(degree_counts) / len(degrees)
    
    # CCDF는 0이 아닌 디그리 값에 대해서만 의미 있음
    non_zero_degrees = np.where(degree_counts > 0)[0]
    if len(non_zero_degrees) == 0:
        print("CCDF를 그릴 디그리 값이 없음")
        return
    
    # CCDF 플롯을 위한 x, y 값 준비
    x_ccdf = non_zero_degrees
    y_ccdf = ccdf[non_zero_degrees]

    plt.figure(figsize=(12, 8))
    plt.loglog(x_ccdf, y_ccdf, marker='o', linestyle='-', color='blue', alpha=0.7, markersize=5)
    
    plt.title('Degree Distribution CCDF', fontsize=18)
    plt.xlabel('Degree (k)', fontsize=14)
    plt.ylabel('P(X >= k)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"디그리 분포 CCDF 그래프가 '{output_filename}' 파일로 저장되었습니다")

def visualize_degree_pdf(G: nx.Graph, output_filename: str):
    """네트워크의 디그리 분포의 확률 밀도 함수(PDF)를 시각화하고 파일로 저장

    Args:
        G (nx.Graph): NetworkX 그래프 객체
        output_filename (str): 저장할 이미지 파일 이름
    """
    if not G.nodes():
        print("PDF를 계산할 노드가 없음")
        return

    degrees = np.array([G.degree(n) for n in G.nodes()])
    
    # 디그리가 0인 노드는 로그 스케일에서 문제가 되므로 제외하거나 처리해야 하지만,
    # 보통 네트워크 분석에서 디그리 분포는 k >= 1에 대해 주로 봄
    degrees = degrees[degrees > 0]
    
    if len(degrees) == 0:
        print("PDF를 그릴 디그리(>0) 값이 없음")
        return

    # PDF 계산 (각 디그리 값의 빈도수 / 전체 노드 수)
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    pdf = counts / len(degrees) # 정규화

    plt.figure(figsize=(12, 8))
    plt.loglog(unique_degrees, pdf, marker='o', linestyle='none', color='green', alpha=0.7, markersize=5)
    
    plt.title('Degree Distribution PDF', fontsize=18)
    plt.xlabel('Degree (k)', fontsize=14)
    plt.ylabel('P(k)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"디그리 분포 PDF 그래프가 '{output_filename}' 파일로 저장되었습니다")
