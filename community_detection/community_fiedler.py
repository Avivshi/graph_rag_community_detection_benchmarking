import numpy as np
import networkx as nx


def fiedler_partition(G):
    """
    G: a NetworkX graph (ideally undirected for a standard Laplacian).
    Returns two lists of nodes: community_1, community_2
    based on sign of Fiedler vector.
    """
    # Build Laplacian
    L = nx.laplacian_matrix(G).astype(float).todense()
    # Compute eigenvalues, eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(L)
    # Sort
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Fiedler vector is the eigenvector for the 2nd smallest eigenvalue
    fiedler_vec = np.array(eigenvectors[:, 1]).flatten()

    # Partition
    nodes = list(G.nodes())
    community_1 = [n for n, val in zip(nodes, fiedler_vec) if val < 0]
    community_2 = [n for n, val in zip(nodes, fiedler_vec) if val >= 0]

    return community_1, community_2
