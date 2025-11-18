import pandas as pd
import networkx as nx
import community as community_louvain

def create_network_from_edges(edges_df: pd.DataFrame, weight_col: str = 'Correlation'):
    """
    엣지 목록 DataFrame으로부터 NetworkX 그래프를 생성합니다.

    :param edges_df: 'ticker1', 'ticker2' 및 가중치 컬럼을 포함하는 DataFrame.
    :param weight_col: 엣지 가중치로 사용할 컬럼 이름.
    :return: NetworkX 그래프 객체.
    """
    G = nx.from_pandas_edgelist(
        edges_df,
        source='ticker1',
        target='ticker2',
        edge_attr=weight_col
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