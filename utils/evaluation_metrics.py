"""
Comprehensive evaluation metrics for community detection algorithms.
Includes both graph-based and embedding-based evaluation methods.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def calculate_modularity(G: nx.Graph, partition_dict: Dict[int, List[str]]) -> float:
    """Calculate modularity score for a partition."""
    # Convert partition dict to list of communities
    communities = list(partition_dict.values())
    
    # Check if partition covers all nodes
    all_partition_nodes = set()
    for community in communities:
        all_partition_nodes.update(community)
    
    graph_nodes = set(G.nodes())
    missing_nodes = graph_nodes - all_partition_nodes
    
    # Intelligently distribute missing nodes
    if missing_nodes:
        if communities:
            communities = _distribute_missing_nodes(G, communities, missing_nodes)
        else:
            communities = [list(missing_nodes)]
    
    # Check for overlapping nodes
    all_nodes_check = []
    for community in communities:
        all_nodes_check.extend(community)
    
    if len(all_nodes_check) != len(set(all_nodes_check)):
        print("Warning: Overlapping nodes detected in partition")
        # Remove duplicates by rebuilding communities
        used_nodes = set()
        clean_communities = []
        for community in communities:
            clean_community = [node for node in community if node not in used_nodes]
            used_nodes.update(clean_community)
            if clean_community:  # Only add non-empty communities
                clean_communities.append(clean_community)
        communities = clean_communities
    
    return nx.community.modularity(G, communities)


def calculate_conductance(G: nx.Graph, partition_dict: Dict[int, List[str]]) -> float:
    """Calculate average conductance across all communities."""
    # Fix partition coverage issues
    communities = list(partition_dict.values())
    
    # Check if partition covers all nodes
    all_partition_nodes = set()
    for community in communities:
        all_partition_nodes.update(community)
    
    graph_nodes = set(G.nodes())
    missing_nodes = graph_nodes - all_partition_nodes
    
    # Intelligently distribute missing nodes
    if missing_nodes:
        if communities:
            communities = _distribute_missing_nodes(G, communities, missing_nodes)
        else:
            communities = [list(missing_nodes)]
    
    conductances = []
    for community_nodes in communities:
        if len(community_nodes) > 0:
            subgraph = G.subgraph(community_nodes)
            edges_within = subgraph.number_of_edges()
            
            # Count edges from community to rest of graph
            edges_outside = 0
            for node in community_nodes:
                for neighbor in G.neighbors(node):
                    if neighbor not in community_nodes:
                        edges_outside += 1
            
            total_edges_from_community = edges_within * 2 + edges_outside  # *2 because within edges are counted twice
            
            if total_edges_from_community > 0:
                conductance = edges_outside / total_edges_from_community
                conductances.append(conductance)
    
    return np.mean(conductances) if conductances else 0.0


def calculate_coverage(G: nx.Graph, partition_dict: Dict[int, List[str]]) -> float:
    """Calculate coverage - fraction of edges within communities."""
    total_edges = G.number_of_edges()
    if total_edges == 0:
        return 0.0
    
    # Fix partition coverage issues
    communities = list(partition_dict.values())
    
    # Check if partition covers all nodes
    all_partition_nodes = set()
    for community in communities:
        all_partition_nodes.update(community)
    
    graph_nodes = set(G.nodes())
    missing_nodes = graph_nodes - all_partition_nodes
    
    # Intelligently distribute missing nodes
    if missing_nodes:
        if communities:
            communities = _distribute_missing_nodes(G, communities, missing_nodes)
        else:
            communities = [list(missing_nodes)]
    
    intra_community_edges = 0
    for community_nodes in communities:
        subgraph = G.subgraph(community_nodes)
        intra_community_edges += subgraph.number_of_edges()
    
    return intra_community_edges / total_edges


def calculate_performance(G: nx.Graph, partition_dict: Dict[int, List[str]]) -> float:
    """Calculate performance metric - fraction of correctly classified node pairs."""
    nodes = list(G.nodes())
    total_pairs = len(nodes) * (len(nodes) - 1) // 2
    if total_pairs == 0:
        return 1.0
    
    # Fix partition coverage issues (same as in calculate_modularity)
    communities = list(partition_dict.values())
    
    # Check if partition covers all nodes
    all_partition_nodes = set()
    for community in communities:
        all_partition_nodes.update(community)
    
    graph_nodes = set(G.nodes())
    missing_nodes = graph_nodes - all_partition_nodes
    
    # Intelligently distribute missing nodes
    if missing_nodes:
        if communities:
            communities = _distribute_missing_nodes(G, communities, missing_nodes)
        else:
            communities = [list(missing_nodes)]
    
    # Create node to community mapping from fixed communities
    node_to_community = {}
    for community_id, community_nodes in enumerate(communities):
        for node in community_nodes:
            if node not in node_to_community:  # Avoid duplicates
                node_to_community[node] = community_id
    
    correct_pairs = 0
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            # Skip if either node is not in the mapping (shouldn't happen now)
            if node1 not in node_to_community or node2 not in node_to_community:
                continue
                
            same_community = node_to_community[node1] == node_to_community[node2]
            connected = G.has_edge(node1, node2)
            
            if (same_community and connected) or (not same_community and not connected):
                correct_pairs += 1
    
    return correct_pairs / total_pairs


def calculate_normalized_cut(G: nx.Graph, partition_dict: Dict[int, List[str]]) -> float:
    """Calculate normalized cut for the partition."""
    # Fix partition coverage issues
    communities = list(partition_dict.values())
    
    # Check if partition covers all nodes
    all_partition_nodes = set()
    for community in communities:
        all_partition_nodes.update(community)
    
    graph_nodes = set(G.nodes())
    missing_nodes = graph_nodes - all_partition_nodes
    
    # Intelligently distribute missing nodes
    if missing_nodes:
        if communities:
            communities = _distribute_missing_nodes(G, communities, missing_nodes)
        else:
            communities = [list(missing_nodes)]
    
    total_cut = 0.0
    
    for community_nodes in communities:
        if len(community_nodes) == 0:
            continue
            
        cut = 0
        for node in community_nodes:
            for neighbor in G.neighbors(node):
                if neighbor not in community_nodes:
                    cut += 1
        
        volume = sum(G.degree(node) for node in community_nodes)
        
        if volume > 0:
            total_cut += cut / volume
    
    return total_cut


def calculate_silhouette_score(node_embeddings: Dict[str, np.ndarray], 
                              partition_dict: Dict[int, List[str]]) -> float:
    """Calculate silhouette score using node embeddings."""
    nodes_with_embeddings = [node for node in node_embeddings.keys() 
                           if node_embeddings[node] is not None]
    
    if len(nodes_with_embeddings) < 2:
        return 0.0
    
    embeddings_matrix = np.array([node_embeddings[node] for node in nodes_with_embeddings])
    
    node_to_community = {}
    for community_id, community_nodes in partition_dict.items():
        for node in community_nodes:
            node_to_community[node] = community_id
    
    labels = [node_to_community.get(node, -1) for node in nodes_with_embeddings]
    
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return 0.0
    
    try:
        return silhouette_score(embeddings_matrix, labels)
    except:
        return 0.0

def calculate_calinski_harabasz_score(node_embeddings: Dict[str, np.ndarray], 
                                    partition_dict: Dict[int, List[str]]) -> float:
    """Calculate Calinski-Harabasz score using node embeddings."""
    nodes_with_embeddings = [node for node in node_embeddings.keys() 
                           if node_embeddings[node] is not None]
    
    if len(nodes_with_embeddings) < 2:
        return 0.0
    
    embeddings_matrix = np.array([node_embeddings[node] for node in nodes_with_embeddings])
    
    node_to_community = {}
    for community_id, community_nodes in partition_dict.items():
        for node in community_nodes:
            node_to_community[node] = community_id
    
    labels = [node_to_community.get(node, -1) for node in nodes_with_embeddings]
    
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return 0.0
    
    try:
        return calinski_harabasz_score(embeddings_matrix, labels)
    except:
        return 0.0

def comprehensive_evaluation(G: nx.Graph, 
                           partition_dict: Dict[int, List[str]], 
                           node_embeddings: Dict[str, np.ndarray],
                           method_name: str = "Unknown") -> Dict[str, Any]:
    """Calculate all evaluation metrics for a partition."""
    from .community_utils import analyze_partition  # Import your existing function
    
    metrics = {
        'method': method_name,
        'num_communities': len(partition_dict),
        'modularity': calculate_modularity(G, partition_dict),
        'conductance': calculate_conductance(G, partition_dict),
        'coverage': calculate_coverage(G, partition_dict),
        'performance': calculate_performance(G, partition_dict),
        'normalized_cut': calculate_normalized_cut(G, partition_dict),
    }
    
    # Add embedding-based metrics if embeddings are available
    if node_embeddings:
        metrics['silhouette_score'] = calculate_silhouette_score(node_embeddings, partition_dict)
        metrics['calinski_harabasz_score'] = calculate_calinski_harabasz_score(node_embeddings, partition_dict)
        
        # Your existing metrics from community_utils
        stats, across_dist = analyze_partition(partition_dict, node_embeddings)
        metrics['intra_pairwise_avg'] = np.mean(stats["intra_pairwise"]) if stats["intra_pairwise"] else 0.0
        metrics['intra_centroid_avg'] = np.mean(stats["intra_centroid"]) if stats["intra_centroid"] else 0.0
        metrics['across_distance'] = across_dist
    
    return metrics

def evaluate_all_methods(G: nx.Graph, 
                        node_embeddings: Dict[str, np.ndarray],
                        partitions_dict: Dict[str, Dict[int, List[str]]]) -> pd.DataFrame:
    """Evaluate all community detection methods and return comparison DataFrame."""
    results = []
    for method_name, partition_dict in partitions_dict.items():
        metrics = comprehensive_evaluation(G, partition_dict, node_embeddings, method_name)
        results.append(metrics)
    
    return pd.DataFrame(results)


def _distribute_missing_nodes(G: nx.Graph, communities: List[List[str]], missing_nodes: set) -> List[List[str]]:
    """
    Intelligently distribute missing nodes across communities based on graph structure.
    
    Strategies:
    1. Assign to community with most connections
    2. If no connections, distribute evenly across communities
    3. Create separate community for isolated nodes if many exist
    """
    if not missing_nodes or not communities:
        return communities
    
    print(f"Warning: {len(missing_nodes)} nodes missing from partition, distributing intelligently")
    
    # Create a copy of communities to modify
    new_communities = [list(community) for community in communities]
    
    # Strategy 1: Assign based on graph connections
    assigned_nodes = set()
    for node in missing_nodes:
        best_community_idx = 0
        max_connections = 0
        
        # Count connections to each existing community
        for i, community in enumerate(new_communities):
            connections = sum(1 for neighbor in G.neighbors(node) if neighbor in community)
            if connections > max_connections:
                max_connections = connections
                best_community_idx = i
        
        # If node has connections, assign to best community
        if max_connections > 0:
            new_communities[best_community_idx].append(node)
            assigned_nodes.add(node)
    
    # Strategy 2: Handle remaining isolated nodes
    remaining_nodes = missing_nodes - assigned_nodes
    
    if remaining_nodes:
        # If many isolated nodes (>10% of total), create separate community
        if len(remaining_nodes) > len(G.nodes()) * 0.1:
            print(f"Creating separate community for {len(remaining_nodes)} isolated nodes")
            new_communities.append(list(remaining_nodes))
        else:
            # Distribute evenly across existing communities
            for i, node in enumerate(remaining_nodes):
                community_idx = i % len(new_communities)
                new_communities[community_idx].append(node)
    
    return new_communities
