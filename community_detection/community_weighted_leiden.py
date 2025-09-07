# community_weighted_leiden.py
import igraph as ig
import leidenalg

def run_leiden_on_weighted_graph(G, nodes):
    """
    G is a weighted NetworkX Graph (undirected).
    'nodes' is the list of node IDs in the same order used in G.
    Returns { community_index: [node_ids] } from a weighted Leiden partition.
    """
    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(nodes))
    ig_graph.vs["name"] = nodes

    # Convert edges with weights to igraph
    for (u, v, data) in G.edges(data=True):
        w = data["weight"]
        i = nodes.index(u)
        j = nodes.index(v)
        ig_graph.add_edge(i, j, weight=w)

    # Weighted modularity
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights=ig_graph.es["weight"]
    )

    result = {}
    for c_idx, member_list in enumerate(partition):
        node_ids = [ig_graph.vs[m]["name"] for m in member_list]
        result[c_idx] = node_ids
    return result
