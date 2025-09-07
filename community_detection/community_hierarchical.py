import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def hierarchical_partition(node_ids, node_embeddings, num_clusters=3):
    """
    node_ids: list of all nodes
    node_embeddings: dict {node_id: np.array([...]) or None}

    Returns { cluster_label: [node_ids] } for the chosen number of clusters.
    """
    valid_nodes = [n for n in node_ids if node_embeddings[n] is not None]
    if not valid_nodes:
        return {}

    # Build matrix
    X = np.array([node_embeddings[n] for n in valid_nodes])
    distance_matrix = pdist(X, metric='cosine')
    linkage_matrix = linkage(distance_matrix, method='ward')

    clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    partition = {}
    for node, cluster_label in zip(valid_nodes, clusters):
        partition.setdefault(cluster_label, []).append(node)

    return partition
