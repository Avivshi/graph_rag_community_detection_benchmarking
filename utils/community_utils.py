# community_utils.py
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer


# Embedding Model Setup
# You may want to load the model just once at import time:
_sbert_model = None

def get_sbert_model(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(model_name)
    return _sbert_model

def embed_text(text: str):
    """
    Encodes text using the loaded SentenceTransformer model.
    """
    model = get_sbert_model()
    return model.encode(text)


# Building the Graph
def build_nx_graph_from_edges(nodes, edges, directed=True):
    """
    Create a NetworkX graph from a list of edges.
    `edges` is [(source_id, target_id), ...].
    If directed=False, create an undirected graph.
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


# Similarity & Distance Helpers
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)

def community_centroid(community_nodes, embeddings):
    """
    Returns the centroid vector of a community.
    `embeddings` is a dict {node_id: np.array(...) or None}
    """
    valid = [n for n in community_nodes if embeddings.get(n) is not None]
    if not valid:
        return None
    dim = len(embeddings[valid[0]])
    c = np.zeros(dim, dtype=float)
    for node in valid:
        c += embeddings[node]
    c /= len(valid)
    return c

def average_intra_community_pairwise_distance(community_nodes, embeddings):
    """
    Computes average pairwise distance within a single cluster.
    """
    valid = [n for n in community_nodes if embeddings.get(n) is not None]
    if len(valid) < 2:
        return 0.0
    dists = []
    for i in range(len(valid)):
        for j in range(i+1, len(valid)):
            d = cosine_distance(embeddings[valid[i]], embeddings[valid[j]])
            dists.append(d)
    return np.mean(dists) if dists else 0.0

def average_node_to_centroid_distance(community_nodes, embeddings):
    """
    Computes the average distance from each node to the centroid of that cluster.
    """
    centroid = community_centroid(community_nodes, embeddings)
    if centroid is None:
        return 0.0
    valid = [n for n in community_nodes if embeddings.get(n) is not None]
    dists = []
    for n in valid:
        dists.append(cosine_distance(embeddings[n], centroid))
    return np.mean(dists) if dists else 0.0

def average_inter_community_pairwise_distance(comm_a, comm_b, embeddings):
    """
    Computes average pairwise distance across two different communities.
    """
    valid_a = [n for n in comm_a if embeddings.get(n) is not None]
    valid_b = [n for n in comm_b if embeddings.get(n) is not None]
    if not valid_a or not valid_b:
        return 0.0
    dists = []
    for na in valid_a:
        for nb in valid_b:
            dists.append(cosine_distance(embeddings[na], embeddings[nb]))
    return np.mean(dists) if dists else 0.0


# Partition Analysis
def analyze_partition(partition_dict, embeddings):
    """
    Given a dict: {cluster_label -> [node_ids]} and a dict of embeddings,
    returns a dict of lists for intra-cluster stats + the overall across-cluster distance.
    """
    import numpy as np

    cluster_labels = sorted(partition_dict.keys())
    results = {
        "intra_pairwise": [],
        "intra_centroid": [],
    }
    across_distances = []

    # Intra-distances
    for c_lbl in cluster_labels:
        c_nodes = partition_dict[c_lbl]
        # Pairwise
        pw_dist = average_intra_community_pairwise_distance(c_nodes, embeddings)
        results["intra_pairwise"].append(pw_dist)
        # Node->centroid
        cent_dist = average_node_to_centroid_distance(c_nodes, embeddings)
        results["intra_centroid"].append(cent_dist)

    # Inter-distances
    for i in range(len(cluster_labels)):
        for j in range(i+1, len(cluster_labels)):
            c1_nodes = partition_dict[cluster_labels[i]]
            c2_nodes = partition_dict[cluster_labels[j]]
            dist = average_inter_community_pairwise_distance(c1_nodes, c2_nodes, embeddings)
            across_distances.append(dist)

    across_avg = np.mean(across_distances) if across_distances else 0.0
    return results, across_avg


# Weighted Graph Building
def build_weighted_graph(nodes, edges, node_embeddings, alpha=0.5):
    """
    Creates an undirected Nx Graph where each edge weight is a blend of:
       alpha * 1 (structural) + (1-alpha) * cos_similarity(emb_u, emb_v).
    """
    Gw = nx.Graph()
    Gw.add_nodes_from(nodes)

    for (u, v) in edges:
        emb_u = node_embeddings.get(u)
        emb_v = node_embeddings.get(v)
        if emb_u is not None and emb_v is not None:
            sim = cosine_similarity(emb_u, emb_v)
        else:
            sim = 0.0
        base = 1.0  # structural presence
        w = alpha*base + (1 - alpha)*sim
        if w > 0:
            Gw.add_edge(u, v, weight=w)

    return Gw
