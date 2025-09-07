# community_leiden.py
import igraph as ig
import leidenalg

def leiden_partition(nodes, edges):
    """
    Runs the Leiden algorithm on an unweighted igraph.
    Returns { community_index: [node_ids] }
    """
    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(nodes))
    ig_graph.vs["name"] = nodes

    # Add edges as undirected or directed; for modularity, typically undirected:
    for src, dst in edges:
        i = nodes.index(src)
        j = nodes.index(dst)
        ig_graph.add_edge(i, j)

    # Standard modularity-based partition
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition
    )

    # Convert result to dict
    result = {}
    for c_idx, member_list in enumerate(partition):
        node_ids = [ig_graph.vs[idx]["name"] for idx in member_list]
        result[c_idx] = node_ids

    return result
