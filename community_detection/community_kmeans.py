# community_kmeans.py
import numpy as np
from sklearn.cluster import KMeans


def kmeans_partition(node_ids, node_embeddings, k=3, random_state=42):
    """
    node_ids: list of node IDs
    node_embeddings: dict {node_id: np.array(...) or None}
    Returns a dict { cluster_label -> [node_ids] } for the nodes that had embeddings.
    """
    valid_nodes = [n for n in node_ids if node_embeddings[n] is not None]
    if not valid_nodes:
        return {}

    X = np.array([node_embeddings[n] for n in valid_nodes])
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)

    partition_dict = {}
    for node_id, lbl in zip(valid_nodes, labels):
        partition_dict.setdefault(lbl, []).append(node_id)

    return partition_dict
