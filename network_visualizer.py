import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def visualize_network(G: nx.Graph, partition: dict, output_filename: str = 'network_visualization.png'):
    """
    커뮤니티가 탐지된 네트워크를 시각화하고 파일로 저장합니다.

    :param G: NetworkX 그래프 객체.
    :param partition: 노드와 커뮤니티 ID 매핑 딕셔너리.
    :param output_filename: 저장할 이미지 파일 이름.
    """
    if not G.nodes():
        print("시각화할 노드가 없습니다.")
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
        edge_colors.append('blue' if corr > 0 else 'red')
        edge_widths.append(abs(corr) * 2) # 상관관계 절댓값에 비례하는 엣지 두께

    # 노드 크기를 디그리(연결 수)에 비례하도록 설정
    degrees = [val for (node, val) in G.degree()]
    node_sizes = [20 + d * 15 for d in degrees] # 기본 크기 + (디그리 * 스케일)

    plt.figure(figsize=(20, 20))

    # 노드, 엣지, 라벨 그리기
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, font_size=6, font_family='Malgun Gothic') # Windows
    # nx.draw_networkx_labels(G, pos, font_size=6, font_family='AppleGothic') # macOS

    num_communities = len(set(partition.values()))
    title = (
        f"Stock Network with Louvain Community Detection\n"
        f"(Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, Communities: {num_communities})"
    )
    plt.title(title, size=25)
    plt.axis('off')
    
    # 이미지 파일로 저장
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"네트워크 시각화가 '{output_filename}' 파일로 저장되었습니다.")