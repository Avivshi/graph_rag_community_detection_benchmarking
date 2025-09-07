# community_spectral.py

import numpy as np
from sklearn.cluster import SpectralClustering

def spectral_partition_from_embeddings(node_ids, node_embeddings, n_clusters=3, gamma=1.0):
    """
    node_ids: list of node IDs
    node_embeddings: dict {node_id: np.array or None}

    We'll build an NxN similarity matrix from embeddings 
    using e.g. RBF kernel or just cosine similarity.
    Then we run scikit-learn SpectralClustering.

    Returns a partition dict { cluster_label -> [node_ids] }
    """
    valid_nodes = [n for n in node_ids if node_embeddings[n] is not None]
    if not valid_nodes:
        return {}

    X = np.array([node_embeddings[n] for n in valid_nodes])

    # Option 1: Let SpectralClustering build its own affinity (RBF or nearest neighbors)
    # Option 2 (below): Build a custom similarity matrix if you prefer e.g. cosine. 
    # We'll do RBF for simplicity:
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='rbf',  # rbf uses the radial basis function kernel
        gamma=gamma,     # influences the RBF kernel width
        assign_labels='kmeans',
        random_state=42
    )
    labels = clustering.fit_predict(X)

    partition_dict = {}
    for node_id, lbl in zip(valid_nodes, labels):
        partition_dict.setdefault(lbl, []).append(node_id)
    return partition_dict
