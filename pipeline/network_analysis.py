import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# From run.py
def calculate_centrality(G: nx.Graph):
    """Calculates the Degree Centrality for each node in the graph."""
    if not G.nodes():
        return {}
    return nx.degree_centrality(G)

# From community_detection.py
def create_network_from_edges(edges_df: pd.DataFrame, weight_col: str = 'Correlation', edge_attrs=None):
    """
    Creates a NetworkX graph from a DataFrame of edges.
    """
    G = nx.from_pandas_edgelist(
        edges_df,
        source='ticker1',
        target='ticker2',
        edge_attr=edge_attrs if edge_attrs else [weight_col]
    )
    return G

# From community_detection.py
def detect_communities(G: nx.Graph, weight_col: str = 'Correlation', random_state: int = None):
    """
    Detects communities in a given graph using the Louvain algorithm.
    """
    partition = community_louvain.best_partition(G, weight=weight_col, random_state=random_state)
    return partition

# From community_detection.py
def calculate_inter_community_correlation(all_edges: pd.DataFrame, partition: dict):
    """
    Calculates the average correlation between communities.
    """
    if not partition:
        return pd.DataFrame()

    all_community_ids = sorted(list(set(partition.values())))
    inter_community_matrix = pd.DataFrame(0.0, index=all_community_ids, columns=all_community_ids)

    edges_with_comm = all_edges.copy()
    edges_with_comm['comm1'] = edges_with_comm['ticker1'].map(partition)
    edges_with_comm['comm2'] = edges_with_comm['ticker2'].map(partition)

    inter_edges = edges_with_comm.dropna(subset=['comm1', 'comm2'])
    inter_edges = inter_edges[inter_edges['comm1'] != inter_edges['comm2']]
    
    if inter_edges.empty:
        return inter_community_matrix

    inter_edges['comm_pair'] = inter_edges.apply(lambda row: tuple(sorted((int(row['comm1']), int(row['comm2'])))), axis=1)
    avg_inter_corr_series = inter_edges.groupby('comm_pair')['Correlation'].mean()

    for (c1, c2), corr_val in avg_inter_corr_series.items():
        inter_community_matrix.loc[c1, c2] = corr_val
        inter_community_matrix.loc[c2, c1] = corr_val
        
    return inter_community_matrix

# From network_visualizer.py
def visualize_network(G: nx.Graph, partition: dict, output_filename: str = 'network_visualization.png'):
    """
    Visualizes the network with communities and saves it to a file.
    """
    if not G.nodes():
        print("시각화할 노드가 없습니다.")
        return

    if partition:
        num_communities = len(set(partition.values()))
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
            colors.append('grey')

    for u, v, data in G.edges(data=True):
        u_comm = partition.get(u)
        v_comm = partition.get(v)
        if u_comm is not None and u_comm == v_comm:
            G.edges[u,v]['layout_weight'] = 1.0
        else:
            G.edges[u,v]['layout_weight'] = 0.05

    k_val = 1.5 / np.sqrt(len(G.nodes()))
    pos = nx.spring_layout(G, weight='layout_weight', k=k_val, iterations=100, seed=42)

    edge_colors = []
    edge_widths = []
    for u, v, data in G.edges(data=True):
        corr = data.get('Correlation', 0)
        edge_colors.append('lightblue' if corr > 0 else 'lightcoral')
        edge_widths.append(abs(corr) * 2)

    degrees = [val for (node, val) in G.degree()]
    node_sizes = [20 + d * 15 for d in degrees]

    plt.figure(figsize=(20, 20))

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
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"네트워크 시각화가 '{output_filename}' 파일로 저장되었습니다.")
