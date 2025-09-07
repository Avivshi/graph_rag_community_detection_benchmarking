import networkx as nx
from networkx.algorithms.community import girvan_newman


def girvan_newman_partition(G, num_communities=2):
    """
    Runs the Girvan–Newman community detection until we have `num_communities`.
    G is a NetworkX graph (undirected recommended).
    Returns { community_label: [node_ids], ... }
    """
    # For best results, use an undirected graph. 
    # If G is directed, consider G.to_undirected().

    # girvan_newman generator yields partitions at each step:
    communities_generator = girvan_newman(G)
    
    # The first partition from girvan_newman is 2 communities,
    # the next is 3, etc. We can iterate until we get `num_communities`.
    partition_iteration = 0
    partition_result = None

    for communities in communities_generator:
        partition_iteration += 1
        if partition_iteration == (num_communities - 1):
            # communities is a tuple of sets: (set_of_nodes_1, set_of_nodes_2, ...)
            partition_result = communities
            break
    
    if not partition_result:
        # If num_communities is bigger than we can achieve, 
        # the last iteration is your best bet.
        partition_result = communities

    # Convert the tuple of sets into { cluster_index: [node_ids] }
    partition_dict = {}
    for i, cset in enumerate(partition_result):
        partition_dict[i] = list(cset)

    return partition_dict
