import numpy as np
from utils.community_utils import analyze_partition


def average_of_list(vals):
    """Calculate the average of a list of values."""
    return np.mean(vals) if vals else 0.0

def create_baseline_partitions(nodes, num_clusters=3, random_seed=42):
    """Create baseline partitions for comparison."""
    baselines = {}
    
    # Single community baseline
    baselines['Single Community'] = {0: nodes}
    
    # Random partition baseline
    np.random.seed(random_seed)
    shuffled_nodes = nodes.copy()
    np.random.shuffle(shuffled_nodes)
    chunk_size = len(nodes) // num_clusters
    random_partition = {}
    for i in range(num_clusters):
        start = i * chunk_size
        end = start + chunk_size if i < num_clusters - 1 else len(nodes)
        random_partition[i] = shuffled_nodes[start:end]
    baselines['Random Partition'] = random_partition
    
    return baselines


def evaluate_baselines(baselines, undirected_G, node_embeddings):
    """Evaluate baseline partitions."""
    print("BASELINE EVALUATION:")
    print("="*60)
    
    baseline_results = {}
    for name, partition in baselines.items():
        try:
            stats, across = analyze_partition(partition, node_embeddings)
            baseline_results[name] = {
                'IntraPW': average_of_list(stats['intra_pairwise']),
                'IntraCent': average_of_list(stats['intra_centroid']),
                'Across': across,
                'num_communities': len(partition)
            }
            print(f"{name} ({len(partition)} communities):")
            print(f"  IntraPW: {baseline_results[name]['IntraPW']:.4f}")
            print(f"  IntraCent: {baseline_results[name]['IntraCent']:.4f}")
            print(f"  Across: {baseline_results[name]['Across']:.4f}")
        except Exception as e:
            print(f"{name}: Error - {e}")
            baseline_results[name] = None
    
    return baseline_results