
def get_algorithm_parameters(random_seed=42):
    """Get standardized algorithm parameters for reproducibility."""
    return {
        'Fiedler': {
            'method': 'spectral_bisection',
            'normalized': True,
            'description': 'Spectral bisection using second smallest eigenvalue of Laplacian'
        },
        'K-Means': {
            'n_clusters': 3,
            'random_state': random_seed,
            'distance_metric': 'cosine',
            'description': 'Lloyd\'s algorithm on semantic embeddings with k=3'
        },
        'Leiden': {
            'resolution': 1.0,
            'random_state': random_seed,
            'n_iterations': -1,
            'description': 'CPM optimization with modularity refinement'
        },
        'Hierarchical': {
            'n_clusters': 3,
            'linkage': 'ward',
            'distance_metric': 'euclidean',
            'description': 'Ward linkage minimizing within-cluster variance'
        },
        'Weighted Leiden': {
            'alpha': 0.5,
            'resolution': 1.0,
            'random_state': random_seed,
            'description': 'Structure-semantic hybrid with α=0.5 weighting'
        },
        'Girvan-Newman': {
            'num_communities': 3,
            'centrality_metric': 'edge_betweenness',
            'description': 'Iterative edge removal by betweenness centrality'
        },
        'Spectral': {
            'n_clusters': 3,
            'gamma': 1.0,
            'kernel': 'rbf',
            'random_state': random_seed,
            'description': 'RBF kernel on embedding similarity matrix'
        }
    }

def display_algorithm_parameters(algorithm_params):
    """Display algorithm parameters in a formatted way."""
    print("ALGORITHM PARAMETERS:")
    print("="*60)
    for method, params in algorithm_params.items():
        print(f"\n{method}:")
        for key, value in params.items():
            if key != 'description':
                print(f"  {key}: {value}")
        print(f"  → {params['description']}")
    print("="*60)
